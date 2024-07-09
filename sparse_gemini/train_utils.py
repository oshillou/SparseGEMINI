from typing import List

import torch
from torch import optim
import gemini
import pynvml
import itertools

from unsupervised_lassonet import LambdaPath
from io_utils import save_state_dict, get_all_models, load_model
from data_utils import HistoryItem

def get_free_gpu():
    qty_gpus=torch.cuda.device_count()
    pynvml.nvmlInit()

    best_gpu=0
    best_memory=0
    for i in range(qty_gpus):
        gpu=pynvml.nvmlDeviceGetHandleByIndex(i)
        info_gpu=pynvml.nvmlDeviceGetMemoryInfo(gpu)

        if info_gpu.free>best_memory:
            best_memory=info_gpu.free
            best_gpu=i

    return f"cuda:{best_gpu}"

def cast_type_device(args, elem, device=None):
    if args.gemini == 'wasserstein':
        elem = elem.double()
    if device is not None:
        elem = elem.to(device)
    return elem

def get_criterion(args):
    if args.gemini=='mmd':
        if args.mode=="ovo":
            return gemini.mmd_ovo
        else:
            return gemini.mmd_ova
    else:
        if args.mode=='ovo':
            return gemini.wasserstein_ovo
        else:
            return gemini.wasserstein_ova

def validate_model(args,dataset,model,criterion,device, lambda_):
    val_loader=dataset.get_loader(batch_size=args.batch_size)

    with torch.no_grad():
        total_loss=0
        for x_batch,D_batch in val_loader:
            x_batch,D_batch=cast_type_device(args,x_batch,device), cast_type_device(args,D_batch,device)

            y_pred=torch.softmax(model(x_batch),dim=1)
            total_loss+=criterion(y_pred,D_batch).item()*len(x_batch)

    return total_loss/len(dataset), lambda_*model.l1_regularisation_skip().item()


def train_model(args, dataset,model,optimiser, device, lambda_=0):

    criterion=get_criterion(args)
    best_val_gemini, best_val_reg=validate_model(args,dataset,model,criterion,device,lambda_)

    epochs_since_best_val=0

    train_loader=dataset.get_loader(batch_size=args.batch_size,shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss=0.0

        for x_batch, D_batch in train_loader:
            x_batch,D_batch=cast_type_device(args,x_batch,device), cast_type_device(args,D_batch,device)

            def closure():
                nonlocal total_loss
                optimiser.zero_grad()

                y_pred=torch.softmax(model(x_batch),dim=1)
                loss=criterion(y_pred,D_batch)

                loss.backward()
                total_loss+=loss.item()*len(x_batch)
                return loss

            optimiser.step(closure)

            with torch.no_grad():
                model.prox(lambda_=lambda_*optimiser.param_groups[0]["lr"])

        total_loss/=len(dataset)

        # Epoch control
        val_gemini,val_reg=validate_model(args, dataset,model,criterion, device, lambda_)

        if val_gemini < (2-args.tol)*best_val_gemini or val_reg<args.tol * best_val_reg:
            best_val_reg=val_reg
            best_val_gemini=val_gemini
            epochs_since_best_val=0
        else:
            epochs_since_best_val+=1

        if args.patience is not None and epochs_since_best_val==args.patience:
            break

    with torch.no_grad():
        reg=model.l1_regularisation_skip().item()

    return HistoryItem(
        lambda_=lambda_,
        train_gemini=-total_loss,
        val_gemini=-val_gemini,
        train_reg=reg,
        val_reg=val_reg,
        selected=model.input_mask().cpu(),
        epochs=epoch
    )

def get_optimiser(name,params,lr):
    if name=="adam":
        return optim.Adam(params,lr=lr)
    elif name=="sgd":
        return optim.SGD(params,lr=lr,momentum=0.9)
    else:
        return optim.Adam(params,lr=lr)

def perform_path(args,dataset, model):

    hist: List[HistoryItem] = []

    print("Initialising every item")
    # Get the device
    device=torch.device(get_free_gpu() if args.use_cuda else 'cpu')
    model=cast_type_device(args,model,device)

    # Get the optimiser for the initial setup
    print("Performing intial step (lambda =0)")
    init_optimiser=get_optimiser(args.init_optim, model.parameters(),args.init_lr)
    hist.append(train_model(args, dataset, model, init_optimiser, device))

    # Save this initial dense model
    save_state_dict(args, model.cpu_state_dict(), "dense_model")

    # Initialise the path of the L1 penalty weight
    path_optimiser=get_optimiser(args.path_optim, model.parameters(),args.path_lr)
    lambda_path=LambdaPath(args.lambda_start, args.lambda_multiplier)

    best_gemini=0
    best_gemini_per_feature= {dataset.get_input_shape(): 0}

    while True:
        print("Starting new iteration: lambda = ",lambda_path.get_lambda())

        # If we are in dynamic mode, we need to update the dataset based on the mask
        if args.dynamic_metric is not None:
            dataset.update_mask(model.input_mask().cpu())

        hist.append(train_model(args,dataset,model,path_optimiser,device, lambda_path.get_lambda()))
        hist[-1].log()

        feature_count=model.selected_count()
        print(f"Currently, the model uses {feature_count} features")

        if hist[-1].val_gemini>best_gemini:
            print("Saving this model that has the best GEMINI")
            best_gemini=hist[-1].val_gemini
            save_state_dict(args, model.cpu_state_dict(), "best_gemini_model")
        if (feature_count in best_gemini_per_feature and hist[-1].val_gemini>best_gemini_per_feature[feature_count]) or (feature_count not in best_gemini_per_feature):
            print(f"Saving this model that has the best GEMINI for {feature_count} features")
            best_gemini_per_feature[feature_count]=hist[-1].val_gemini
            save_state_dict(args,model.cpu_state_dict(), f"best_model_{feature_count}_features")

        if feature_count<=args.feature_threshold or feature_count==0:
            break
        lambda_path.next()

    return hist

def get_model_predictions(args,dataset,model,device):
    val_loader=dataset.get_loader(batch_size=args.batch_size)

    if args.dynamic_metric is not None:
        dataset.reset_mask()
    all_predictions=[]
    with torch.no_grad():
        for x_batch,D_batch in val_loader:
            x_batch=cast_type_device(args,x_batch,device)

            y_pred=torch.softmax(model(x_batch),dim=1)

            all_predictions+=[y_pred.argmax(1)]

    return model.l1_regularisation_skip().item(), torch.concat(all_predictions)

def build_clustering_results(args, model, dataset, history: List[HistoryItem]):
    # First associate the lambda value that matched the best model
    features_to_lambda=dict()
    features_to_gemini=dict()
    for hist in history:
        num_features=hist.selected.sum().item()
        if num_features not in features_to_lambda:
            features_to_lambda[num_features]=hist.lambda_
            features_to_gemini[num_features]=hist.val_gemini
        elif hist.val_gemini>features_to_gemini[num_features]:
            features_to_lambda[num_features]=hist.lambda_
            features_to_gemini[num_features]=hist.val_gemini
    all_results=[]

    device=next(model.parameters()).device
    for filename in get_all_models(args):
        model=load_model(args,filename,model)
        num_features=model.selected_count()
        print(f"Working on {num_features}")
        l1,predictions=get_model_predictions(args,dataset,model,device)
        tmp_result={f"c_{i}": predictions[i].cpu().item() for i in range(len(predictions))}
        tmp_result["Loss"]=features_to_gemini[num_features]
        tmp_result["lambda"]=features_to_lambda[num_features]
        tmp_result['l1']=l1
        tmp_result['K']=len(torch.unique(predictions))
        tmp_result["F"]=num_features
        all_results+=[tmp_result]

    return all_results

def compute_feature_importances(path: List[HistoryItem]):
    current = path[0].selected.clone()
    ans = torch.full(current.shape, float("inf"))
    for save in itertools.islice(path, 1, None):
        lambda_ = save.lambda_
        diff = current & ~save.selected
        ans[diff.nonzero().flatten()] = lambda_
        current &= save.selected
    return ans

def build_selection_history(path: List[HistoryItem]):
    all_selection=list(map(lambda x: x.selected, path))

    return torch.cat(all_selection).reshape(len(all_selection),-1)
