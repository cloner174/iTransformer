import os
import time
import torch
import pandas as pd
from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from .pre_train import SaveArgs, load_args


def predict(args, model, predict_root = None, predict_data = None, retrain = False, new_data = None):#model= setting or actual model
    """
    Use Model To Predict Future Days!
    Argumans:
        args: Object | str, The models setup. Can be an Object of type DotDict class, or the path to saved file of it -> (args.json).
        model: str|Object, Whether can be the setting or folder name of path to the 'checkpoint.pth' or the actual model object!
        days_to_predict: int, How much days, should to be predicted!
        predict_data: the name of predict data inside pred folder. if None, will use the current name in args.
        
        retrain: bool, Optional. If True, and new_data is not None, It would change the setting and args to retrain the current model with new data.
        new_data: str, The new data name inside the root path from args. If None, and retrain is True, It would use the current root path and data name in args to retrain model.
                \Will Raise an Error, If no data is available/
    """
    
    if isinstance(args, str):
        try:
            arg = load_args(args)
        except Exception as e:
            raise AssertionError(f"Fail to read args.pkl reason -> {e}")
    else:
        if predict_data is None and new_data is None:
            arg = args
        else:
            try:
                args_path = SaveArgs(args=args, path='', temporary=True)
                args_path = args_path.path
                arg = load_args(args_path)
                os.unlink(args_path)
            except Exception as e:
                raise AssertionError(f"Fail to read args.pkl reason -> {e}")
    
    if retrain and new_data is not None:
        arg.data_path = new_data
    
    if predict_data is not None:
        if predict_root is not None:
            arg.pred_root_path = predict_root
        arg.pred_data_path = predict_data
    
    if isinstance(model, Exp_Long_Term_Forecast) or isinstance(model, Exp_Long_Term_Forecast_Partial):
        if predict_data is None and new_data is None:
            exp = model
        else:
            model.args = arg
            exp = model
    elif isinstance(model, str):
        if arg.exp_name == 'partial_train':
            Exp = Exp_Long_Term_Forecast_Partial
        else:
            Exp = Exp_Long_Term_Forecast
        exp = Exp(arg)
        try:
            path = os.path.join(arg.checkpoints, model)
            if not model.endswith('.pth') or not model.endswith('.pt'):
                path = path + '/' + 'checkpoint.pth'
            
            exp.model.load_state_dict(torch.load(path))
        except Exception as e:
            try:
                if not model.endswith('.pth') or not model.endswith('.pt'):
                    path = model + '/' + 'checkpoint.pth'
                
                exp.model.load_state_dict(torch.load(model))
            except:
                raise AssertionError(f" There was an Error loading your model with the provded path.Assumed path is {model} and Error was: {e}")
    else:
        raise TypeError(" The Model Object can be of type str(model checkpoint.pth path) or the actual model from experiments kind of models from this repo.")
    
    if retrain:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for ii in range(arg.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
                    arg.model_id,
                    arg.model,
                    arg.data,
                    arg.features,
                    arg.seq_len,
                    arg.label_len,
                    arg.pred_len,
                    arg.d_model,
                    arg.n_heads,
                    arg.e_layers,
                    arg.d_layers,
                    arg.d_ff,
                    arg.factor,
                    arg.embed,
                    arg.distil,
                    arg.des,
                    arg.class_strategy, ii, timestamp)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
    
    try:
        df_temp = pd.read_csv(os.path.join(arg.pred_root_path, arg.pred_data_path))
        end_at_first = df_temp.shape[0] - 1
    except:
        print(f'please inter the path to your prediction data in input arguman : predict_root  and  predict_data')
        print('Where predict_root is the main folder contained your csv file and predict_data is name of the csv file with .csv at the end')
        return 0
    
    folder_path = 'results/Prediction Results/'
    os.makedirs(folder_path, exist_ok=True)
    file_path = folder_path + 'prediction.csv'
    
    if os.path.exists(file_path):
        base, ext = os.path.splitext(file_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = f"{base}_{timestamp}{ext}"
    
    pred = exp.predict(setting=arg, load=False, return_=True)
    preds = []
    for i in range(arg.pred_len):
        preds.append( list(pred[0,i,:]) )
    
    cols = list(df_temp.columns)
    date_name = arg.name_of_col_with_date if hasattr(arg, 'name_of_col_with_date') else 'date'
    target = arg.target
    date_index = cols.index(date_name)
    cols.pop(date_index)
    border1 = len(df_temp) - arg.seq_len
    border2 = len(df_temp)
    tmp_stamp = df_temp[[date_name]][border1:border2]
    tmp_stamp[date_name] = pd.to_datetime(tmp_stamp[date_name])
    pred_dates = list( pd.date_range(tmp_stamp.date.values[-1], periods=arg.pred_len + 1, freq=arg.freq) )
    
    for j in range(len(pred_dates)):
        if j+1 == len(pred_dates):
            break
        else:
            next_day = pred_dates[j+1]
        temp = {}
        for i in range(len(cols)):
            col = cols[i]
            if col == target :
                if arg.features == 'MS' or arg.features == 'S' :
                    temp[col] = preds[j][-1]
                else:
                    temp[col] = preds[j][i]
            else:
                if arg.features == 'S':
                    temp[col] = 0
                else:
                    temp[col] = preds[j][i] 
        temp = pd.DataFrame(temp, index=[df_temp.shape[0]])
        temp.insert(loc = date_index, column=date_name, value=next_day)
        df_temp = pd.concat([df_temp, temp])
    
    if arg.features == 'S' or arg.features == 'MS':
            df_temp = pd.concat( [df_temp.loc[end_at_first+1:,date_name], df_temp.loc[end_at_first+1:,target]],axis=1)
    else:
        df_temp = df_temp.loc[end_at_first:,:]
    df_temp.to_csv(file_path, index = False)
    print(f'''The Results of Prediction for The Next {arg.pred_len} Days Are Now Stored in 
                {file_path}''')
    return True