from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.criter import WeightedMeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError, RMSELoss,QuantileLoss, HuberLoss, PinballLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from .pre_train import SaveArgs

warnings.filterwarnings('ignore')

print("This is The enhanced version of Orginal code, Written in 2024")
time.sleep(1)
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_losses = []
        self.test_losses = []
        #self.vali_losses = []
        #self.trues_during_vali = []
        #self.preds_during_vali = []
        if args.is_training != 0:
            try:
                SaveArgs(args=args, path='input')
            except:
                print("Fail To Save The Args. Continue ..")
            time.sleep(1)
        
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.args.kind_of_optim == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'SparseAdam':
            model_optim = optim.SparseAdam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'RMSprop':
            model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'RAdam':
            model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'NAdam':
            model_optim = optim.NAdam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'LBFGS':
            model_optim = optim.LBFGS(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'Adamax':
            model_optim = optim.Adamax(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'ASGD':
            model_optim = optim.ASGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'Adadelta':
            model_optim = optim.Adadelta(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.kind_of_optim == 'Adagrad':
            model_optim = optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim

    def _select_criterion(self):
        if self.args.criter.lower() == 'wmape':
            criterion = WeightedMeanAbsolutePercentageError()
        elif self.args.criter.lower() == 'smape':
            criterion = SymmetricMeanAbsolutePercentageError()
        elif self.args.criter.lower() == 'mae':
            criterion = nn.L1Loss()
        elif self.args.criter.lower() == 'rmse':
            criterion = RMSELoss()
        elif self.args.criter.lower() == 'quantileloss':
            criterion = QuantileLoss()
        elif self.args.criter.lower() == 'huberloss':
            criterion = HuberLoss()
        elif self.args.criter.lower() == 'pinballloss':
            criterion = PinballLoss()
        else:
            criterion = nn.MSELoss()  # Default to Mean Squared Error
        
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        #trues_during_vali = []
        #preds_during_vali = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                #trues_during_vali.append(batch_y.detach().cpu().numpy())
                #preds_during_vali.append(outputs.detach().cpu().numpy())
                
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        #try:
            #if len(self.trues_during_vali) == 0:
                #trues_during_vali = np.array(trues_during_vali)
                #preds_during_vali = np.array(preds_during_vali)
                #self.trues_during_vali = trues_during_vali.reshape(-1, trues_during_vali.shape[-2], trues_during_vali.shape[-1])
                #self.preds_during_vali = preds_during_vali.reshape(-1, preds_during_vali.shape[-2], preds_during_vali.shape[-1])
            #else:
                #shape_self_true = self.trues_during_vali.shape
                #shape_self_pred = self.preds_during_vali.shape
            
                #trues_during_vali = np.array(trues_during_vali)
                #preds_during_vali = np.array(preds_during_vali)
                #trues_during_vali = trues_during_vali.reshape(-1, trues_during_vali.shape[-2], trues_during_vali.shape[-1])
                #preds_during_vali = preds_during_vali.reshape(-1, preds_during_vali.shape[-2], preds_during_vali.shape[-1])
                #shape_funv_true = trues_during_vali.shape
                #shape_funv_pred = preds_during_vali.shape
            
                #self.trues_during_vali = self.trues_during_vali.flatten().tolist()
                #self.preds_during_vali = self.preds_during_vali.flatten().tolist()
                #trues_during_vali = trues_during_vali.flatten().tolist()
                #preds_during_vali = preds_during_vali.flatten().tolist()
                #trues_during_vali = self.trues_during_vali + trues_during_vali
                #preds_during_vali = self.preds_during_vali + preds_during_vali
            
                #trues_during_vali = np.array(trues_during_vali)
                #preds_during_vali = np.array(preds_during_vali)
                #self.trues_during_vali = trues_during_vali.reshape(shape_funv_true[0]+shape_self_true[0], shape_self_true[1], shape_self_true[2])
                #self.preds_during_vali = preds_during_vali.reshape(shape_self_pred[0]+shape_funv_pred[0], shape_self_pred[1],shape_self_pred[2])
        #except:    
            #pass
        return total_loss
    
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        #vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        trues_during_training = []
        preds_during_training = []
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    preds_during_training.append(outputs.detach().cpu().numpy())
                    trues_during_training.append(batch_y.detach().cpu().numpy())
                    train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            #self.vali_losses.append(vali_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        preds_during_training = np.array(preds_during_training)
        trues_during_training = np.array(trues_during_training)
        print('\n')
        print('train shape:', preds_during_training.shape, trues_during_training.shape)
        preds_during_training = preds_during_training.reshape(-1, preds_during_training.shape[-2], preds_during_training.shape[-1])
        trues_during_training = trues_during_training.reshape(-1, trues_during_training.shape[-2], trues_during_training.shape[-1])
        print('train shape:', preds_during_training.shape, trues_during_training.shape)
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds_during_training, trues_during_training)
        
        print('Train mse:{},Train mae:{}'.format(mse, mae))
        print('Train rmse:{},Train mape:{}'.format(rmse, mape))
        print('\n')
        time.sleep(2)
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Train mse:{},Train mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics_during_training.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'preds_during_training.npy', preds_during_training)
        np.save(folder_path + 'trues_during_training.npy', trues_during_training)
        #try:
            #preds_during_vali = np.array(self.preds_during_vali)
            #trues_during_vali = np.array(self.trues_during_vali)
            #print('Validate shape:', (preds_during_vali.shape[0]//self.args.batch_size, self.args.batch_size, preds_during_vali.shape[1],preds_during_vali.shape[2]),(trues_during_vali.shape[0]//self.args.batch_size, self.args.batch_size, trues_during_vali.shape[1],trues_during_vali.shape[2]))
            #preds_during_vali = preds_during_vali.reshape(-1, preds_during_vali.shape[-2], preds_during_vali.shape[-1])
            #trues_during_vali = trues_during_vali.reshape(-1, trues_during_vali.shape[-2], trues_during_vali.shape[-1])
            #print('Validate shape:', preds_during_vali.shape, trues_during_vali.shape)
        
            #mae, mse, rmse, mape, _ = metric(preds_during_vali, trues_during_vali)
            #print('Validate mse:{},Validate mae:{}'.format(mse, mae))
            #print('Validate rmse:{},Validate mape:{}'.format(rmse, mape))
            #print('\n')
            #time.sleep(2)
            #f = open("result_long_term_forecast.txt", 'a')
            #f.write("Validate Info:" + "  \n")
            #f.write('mse:{}, mae:{}'.format(mse, mae))
            #f.write('\n')
            #f.write('\n')
            #f.close()
        #except:
            #pass
        return self.model
    
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                
                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if self.args.do_visual:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('Test mse:{},Test mae:{}'.format(mse, mae))
        print('Test rmse:{},Test mape:{}'.format(rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                self.batch_y = batch_y
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)
        
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        pred_save_path = folder_path + 'Preds real_prediction.npy'
        np.save(folder_path + 'Preds real_prediction.npy', preds)
        
        print(f'''The Results of Prediction for The Next {self.args.pred_len} Days Are Now Stored in 
                {pred_save_path}''')
        return
