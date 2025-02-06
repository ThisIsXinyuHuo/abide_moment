from datetime import datetime
from pathlib import Path
import sys
from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from download_abide_dataset.load_abide import get_dataset
from joblib import dump

import argparse
from argparse import Namespace
import random
import numpy as np
import os 
import pdb
import logging

logger = logging.getLogger(__name__)



def set_seed(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ABIDE_Trainer:
    def __init__(self, args: Namespace):
        self.args = args

        # set up logger
        if "SLURM_JOB_ID" in os.environ:
            self.output_dir = Path(
                f"{self.args.output_dir}/{os.environ['SLURM_JOB_ID']}"
            )
        else:
            self.output_dir = Path(
                f"{self.args.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "training.log"


        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path)
            ],
        )

        log_level = logging.INFO # logging.INFO, "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        logger.setLevel(log_level)

        

        #initialize ABIDE classification dataset 
        dataset = get_dataset(dataset_path = self.args.dataset_path,
                                   data_dir= self.args.data_dir,
                                   trust_remote_code = self.args.trust_remote_code,
                                   validation=self.args.validation, 
                                   seed=self.args.seed)
        
        self.train_dataset = dataset["train"] 
        if self.args.max_train_samples is not None:
            max_train_samples = min(len(self.train_dataset), self.args.max_train_samples)
            self.train_dataset = self.train_dataset.select(range(max_train_samples))
        self.train_dataset.set_transform(self.transform_images)
        logger.info(f"Training set size: {len(self.train_dataset)}")

        self.val_dataset = dataset["validation"] if self.args.validation else None
        if self.args.validation:
            if self.args.max_val_samples is not None:
                max_val_samples = min(len(self.val_dataset), self.args.max_val_samples)
                self.val_dataset = self.val_dataset.select(range(max_val_samples))
            self.val_dataset.set_transform(self.transform_images)
            logger.info(f"Validation set size: {len(self.val_dataset)}")


        self.test_dataset = dataset["test"]
        if self.args.max_test_samples is not None:
            max_test_samples = min(len(self.test_dataset), self.args.max_test_samples)
            self.test_dataset = self.test_dataset.select(range(max_test_samples))
        self.test_dataset.set_transform(self.transform_images)
        logger.info(f"Testing set size: {len(self.test_dataset)}")

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, generator=generator)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, generator=generator)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False, generator=generator) if self.args.validation else None
        print("Dataloader created!")

        #linear probing: only train classification head
        #finetuning: train both encoder and classification head
        #unsupervised learning: train SVM on top of MOMENT embeddings
        self.model = MOMENTPipeline.from_pretrained(
                                                    f"./models/MOMENT-1-{self.args.model}",
                                                    model_kwargs={
                                                        'task_name': 'classification',
                                                        'n_channels': 200,
                                                        'num_class': 2,
                                                        'freeze_encoder': False if self.args.mode == 'full_finetuning' else True,
                                                        'freeze_embedder': False if self.args.mode == 'full_finetuning' else True,
                                                        'reduction': self.args.reduction,
                                                        #Disable gradient checkpointing for finetuning or linear probing to 
                                                        #avoid warning as MOMENT encoder is frozen
                                                        'enable_gradient_checkpointing': True if self.args.mode == 'full_finetuning' else False, 
                                                    },
                                                    local_files_only=True
                                                )
        self.model.init()
        print('Model initialized, training mode: ', self.args.mode)

        #using cross entropy loss for classification 
        self.criterion = torch.nn.CrossEntropyLoss()
        
        if self.args.mode == 'full_finetuning':
            print('Encoder and embedder are trainable')
            if self.args.lora:
                lora_config = LoraConfig(
                                        r=64,
                                        lora_alpha=32,
                                        target_modules=["q", "v"],
                                        lora_dropout=0.05,
                                        )
                self.model = get_peft_model(self.model, lora_config)
                print('LoRA enabled')
                self.model.print_trainable_parameters()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, 
                                                            total_steps=self.args.epochs*len(self.train_dataloader))
            
            #set up model ready for accelerate finetuning
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)
        
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, 
                                                            total_steps=self.args.epochs*len(self.train_dataloader))
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        
        print(f"ASD classification training, mode: {self.args.mode}\n")
        logger.info(f"ASD classification training, mode: {self.args.mode}\n")

    def transform_images(self, batch):
        time_series_lst = [np.loadtxt(
            time_series_path, dtype=np.float32
        ) for time_series_path in batch['time_series_path']] # bs x sequence_length x num_input_channels

        bs = len(time_series_lst)
        #print(bs)
        sequence_length = 512
        num_input_channels = time_series_lst[0].shape[-1]

        mask = np.zeros((bs, sequence_length, num_input_channels), dtype=np.bool_)

        for i in range(len(time_series_lst)):
            time_series = time_series_lst[i]
            # truncate
            if time_series.shape[0] > sequence_length:
                time_series = time_series[:sequence_length]
            # mask
            mask[i, :time_series.shape[0]] = 1
            # pad
            time_series_lst[i] = np.pad(
                time_series, ((0, sequence_length - time_series.shape[0]), (0, 0))
            )
        time_series_lst = np.stack(time_series_lst, axis=0).transpose(0,2,1)
        #print(time_series_lst.shape)
        mask = mask.transpose(0,2,1)[:, 0, :]

        batch['time_series'] = torch.from_numpy(time_series_lst)
        batch['mask'] = torch.from_numpy(mask)
        return batch
    
    def get_embeddings(self, dataloader: DataLoader):
        '''
        labels: [num_samples]
        embeddings: [num_samples x d_model]
        '''
        embeddings, labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch_x, batch_mask, batch_labels = batch["time_series"], batch["mask"], batch["label"]
                # [batch_size x 200 x 512], [batch_size x 512], [batch_size]
                batch_x = batch_x.to(self.device).float()
                batch_mask = batch_mask.to(self.device).float()
                # [batch_size x num_patches x d_model (=1024)]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x, input_mask = batch_mask, reduction=self.args.reduction) 
                # mean over patches dimension, [batch_size x d_model]
                embedding = output.embeddings.mean(dim=1)
                embeddings.append(embedding.detach().cpu().numpy())
                labels.append(batch_labels)        

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        return embeddings, labels
    
    def get_timeseries(self, dataloader: DataLoader, agg='mean'):
        '''
        mean: average over all channels, result in [1 x seq_len] for each time-series
        channel: concat all channels, result in [1 x seq_len * num_channels] for each time-series

        labels: [num_samples]
        ts: [num_samples x seq_len] or [num_samples x seq_len * num_channels]
        '''
        ts, labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch_x, batch_mask, batch_labels = batch["time_series"], batch["mask"], batch["label"]
                # [batch_size x 12 x 512]
                if agg == 'mean':
                    batch_x = batch_x.mean(dim=1)
                    ts.append(batch_x.detach().cpu().numpy())
                elif agg == 'channel':
                    ts.append(batch_x.view(batch_x.size(0), -1).detach().cpu().numpy())
                labels.append(batch_labels)        

        ts, labels = np.concatenate(ts), np.concatenate(labels)
        return ts, labels
    
    def train(self):
        for epoch in range(self.args.epochs):

            print(f'Epoch {epoch+1}/{self.args.epochs}')
            logger.info(f'Epoch {epoch+1}/{self.args.epochs}\n')
        
            self.epoch = epoch + 1

            if self.args.mode == 'linear_probing':
                self.train_epoch_lp()
                self.evaluate_epoch()
                self.save_checkpoint(epoch = epoch)
            
            elif self.args.mode == 'full_finetuning':
                self.train_epoch_ft()
                self.evaluate_epoch()
                self.save_checkpoint(epoch = epoch)
            
            #break after training SVM, only need one 'epoch'
            elif self.args.mode == 'unsupervised_representation_learning':
                self.train_ul()
                break

            elif self.args.mode == 'svm':
                self.train_svm()
                break

            else:
                raise ValueError('Invalid mode, please choose svm, linear_probing, full_finetuning, or unsupervised_representation_learning')

#####################################training loops#############################################
    def train_epoch_lp(self):
        '''
        Train only classification head
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            batch_x, batch_mask, batch_labels = batch["time_series"], batch["mask"], batch["label"]
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_mask = batch_mask.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, input_mask = batch_mask, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())
            torch.cuda.empty_cache()
            print(" ")
        
        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        logger.info(f'Train loss: {avg_loss}\n')
    
    def train_epoch_ft(self):
        '''
        Train encoder and classification head (with accelerate enabled)
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            batch_x, batch_mask, batch_labels = batch["time_series"], batch["mask"], batch["label"]
            self.optimizer.zero_grad()
            batch_x = batch_x.to(self.device).float()
            batch_mask = batch_mask.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(x_enc=batch_x, input_mask = batch_mask, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
                losses.append(loss.item())
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()
            print(" ")

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        logger.info(f'Train loss: {avg_loss}\n')
       
    
    def train_ul(self):
        '''
        Train SVM on top of MOMENT embeddings
        '''
        self.model.eval()
        self.model.to(self.device)

        #extract embeddings and label
        train_embeddings, train_labels = self.get_embeddings(self.train_dataloader)
        print('embedding shape: ', train_embeddings.shape)
        print('label shape: ', train_labels.shape)

        #fit statistical classifier
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        self.save_svm()
        print('Train accuracy: ', train_accuracy)
        logger.info(f'Train accuracy: {train_accuracy}\n')

    def train_svm(self):
        '''
        Train SVM on top of timeseries data
        '''
        train_embeddings, train_labels = self.get_timeseries(self.train_dataloader, agg=self.args.agg)
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        self.save_svm()
        print('Train accuracy: ', train_accuracy)
        logger.info(f'Train accuracy: {train_accuracy}\n')
#####################################training loops#################################################

#####################################evaluate loops#################################################
    def test(self):
        if self.args.mode == 'unsupervised_representation_learning':
            test_embeddings, test_labels = self.get_embeddings(self.test_dataloader)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            logger.info(f'Test accuracy: {test_accuracy}\n')

        elif self.args.mode == 'linear_probing' or self.args.mode == 'full_finetuning':
            self.evaluate_epoch(phase='test')

        elif self.args.mode =='svm':
            test_embeddings, test_labels = self.get_timeseries(self.test_dataloader, agg=self.args.agg)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            logger.info(f'Test accuracy: {test_accuracy}\n')

        else:
            raise ValueError('Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')
        
    def evaluate_epoch(self, phase='val'):
        if phase == 'val':
            dataloader = self.val_dataloader
        elif phase == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.model.eval()
        self.model.to(self.device)
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch_x, batch_mask, batch_labels = batch["time_series"], batch["mask"], batch["label"]
                batch_x = batch_x.to(self.device).float()
                batch_mask = batch_mask.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(x_enc=batch_x, input_mask = batch_mask, reduction=self.args.reduction)
                    loss = self.criterion(output.logits, batch_labels)
                total_loss += loss.item()
                total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader.dataset)
        print(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}')
        logger.info(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}\n')
#####################################evaluate loops#################################################

    def save_checkpoint(self, epoch = None):
        if self.args.mode in ['svm', 'unsupervised_representation_learning']:
            raise ValueError('No checkpoint to save for SVM or unsupervised learning, as no training was done')
        
        

        directory = Path(f"{self.output_dir}/saved_models_epoch_{epoch}") if epoch else Path(f"{self.output_dir}/saved_models_final")

        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), directory/f"MOMENT_Classification.pth")
        print('Model saved at', directory)
        
        model1 = MOMENTPipeline.from_pretrained(
                                                    f"./models/MOMENT-1-{self.args.model}",
                                                    model_kwargs={
                                                        'task_name': 'classification',
                                                        'n_channels': 200,
                                                        'num_class': 2,
                                                        'freeze_encoder': False if self.args.mode == 'full_finetuning' else True,
                                                        'freeze_embedder': False if self.args.mode == 'full_finetuning' else True,
                                                        'reduction': self.args.reduction,
                                                        #Disable gradient checkpointing for finetuning or linear probing to 
                                                        #avoid warning as MOMENT encoder is frozen
                                                        'enable_gradient_checkpointing': True if self.args.mode == 'full_finetuning' else False, 
                                                    },
                                                    local_files_only=True
                                                )
        model1.init()
        model1.load_state_dict(torch.load(directory/f"MOMENT_Classification.pth", weights_only=True))
        model2 = MOMENTPipeline.from_pretrained(
                                                    f"./models/MOMENT-1-{self.args.model}",
                                                    model_kwargs={
                                                        'task_name': 'classification',
                                                        'n_channels': 200,
                                                        'num_class': 2,
                                                        'freeze_encoder': False if self.args.mode == 'full_finetuning' else True,
                                                        'freeze_embedder': False if self.args.mode == 'full_finetuning' else True,
                                                        'reduction': self.args.reduction,
                                                        #Disable gradient checkpointing for finetuning or linear probing to 
                                                        #avoid warning as MOMENT encoder is frozen
                                                        'enable_gradient_checkpointing': True if self.args.mode == 'full_finetuning' else False, 
                                                    },
                                                    local_files_only=True
                                                )
        model2.init()
        model2.load_state_dict(torch.load(directory/f"MOMENT_Classification.pth", weights_only=True))
        print(compare_models(model1, model2))
        

    def save_svm(self):
        directory = Path(
            f"{self.output_dir}/saved_models"
        )
        directory.mkdir(parents=True, exist_ok=True)
        dump(self.clf, directory/"svm_model.joblib")

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            #print( key_item_1[0])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
    
if __name__ == '__main__':
    


    parser = argparse.ArgumentParser()
    #training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mode', type=str, default='full_finetuning', help='choose from linear_probing, full_finetuning, unsupervised_representation_learning')
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--agg', type=str, default='channel', help='aggregation method for timeseries data for svm training, choose from mean or channel')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora', action='store_true', help='enable LoRA')
    parser.add_argument('--reduction', type=str, default='concat', help='reduction method for MOMENT embeddings, choose from mean or max')
    parser.add_argument('--model', type=str, default="large")
    #abide dataset parameters
    parser.add_argument('--dataset_path', type=str, help='path to load ABIDE dataset')
    parser.add_argument('--data_dir', type=str, help='dir of ABIDE dataset')
    parser.add_argument('--trust_remote_code', action='store_true', help="enable validation")
    parser.add_argument('--validation', action='store_true', help="enable validation")
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    #output
    parser.add_argument('--output_dir', type=str)

    
    args = parser.parse_args()
    set_seed(args.seed)

    trainer = ABIDE_Trainer(args)
    #trainer.train()
    #trainer.test()
    trainer.save_checkpoint()
        
        

