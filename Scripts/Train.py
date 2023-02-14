import numpy
import torch
import time
import numpy as np
from tqdm.notebook import tqdm
from Metrics import pixel_accuracy, meanIoU
from Configuration import Configs



def get_lr(optimizer):
    '''
    Return the learning rate value of optimizer
    '''
  for group in optimizer.param_groups: 
    return group['lr']



def fit(epochs:int, model, device, train_loader, val_loader, criterion, optimizer, scheduler, patch:bool = False):
    '''
    Setting a training loop.
    Returns dictionary with results.
    '''
    
    torch.cuda.empty_cache()
    
    #Training information to save from every loop
    train_losses     = []
    valid_losses     = []
    valid_IoUs       = []
    valid_accuracies = []
    train_IoUs       = []
    train_accuracies = []
    learn_rates      = []
    
    #Paramters being updated through training/validation
    min_loss     = np.inf
    decrease     = 1
    not_improve  = 0

    #configurations
    model.to(device)
    fit_time = time.time()
    
    
    print('Training started \n \n') 
    
    #Training loop
    for epoch in range(epochs):
        since = time.time()
        running_loss = 0
        IoU_score    = 0
        accuracy     = 0
        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles  = mask_tiles.view(-1, h, w)
            
            image = image_tiles.to(device)
            mask  = mask_tiles.to(device)
            
            #forward
            output = model(image)
            loss   = criterion(output, mask)
            print(loss)
            
            #evaluation metrics
            IoU_score += meanIoU(output, mask, Configs.num_classes, smooth = False)
            accuracy  += pixel_accuracy(output, mask)
            
            #backward
            loss.backward()
            optimizer.step()      #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            learn_rates.append(get_lr(optimizer))
            scheduler.step()
            
            running_loss += loss.item()
            
        else:
            
            #Validation
            print('validation')
            model.eval()
            valid_loss      = 0
            valid_accuracy  = 0
            valid_IoU_score = 0

            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data
                    
                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles  = mask_tiles.view(-1, h, w)
                    
                    image  = image_tiles.to(device)
                    mask   = mask_tiles.to(device)

                    output = model(image)

                    #evaluation metrics
                    valid_IoU_score += meanIoU(output, mask)
                    valid_accuracy  += pixel_accuracy(output, mask)
                    #loss
                    loss = criterion(output, mask)                                  
                    valid_loss += loss.item()
            
            #calculate mean for each batch
            train_losses.append(running_loss / len(train_loader))
            valid_losses.append(valid_loss / len(val_loader))


            if min_loss > (valid_loss / len(val_loader)):
                print('Loss Decreased by {:.3f}'.format(min_loss -  (valid_loss / len(val_loader))))
                min_loss = (valid_loss / len(val_loader))
                decrease += 1
                if decrease % 2 == 0:
                    print('saving model...')
                    torch.save(model, f'{Configs.model_folder}{Configs.model.name}.pt')


            if min_loss < (valid_loss / len(val_loader)):
                not_improve += 1
                min_loss = (valid_loss / len(val_loader))
                print(f'Loss Not Decreased for {not_improve} times')
                if not_improve == 7:
                    print('Loss Not Decreased for 7 times, Training Stoped ')
                    break
            
            #Metrics
            valid_IoUs.append(valid_IoU_score / len(val_loader))
            train_IoUs.append(IoU_score / len(train_loader))
            train_accuracies.append(accuracy / len(train_loader))
            valid_accuracies.append(valid_accuracy / len(val_loader))
            
            print("Epoch:{}/{}   |"          .format(epoch + 1, epochs),
                  "Train Loss: {:.3f}   |"   .format(running_loss   /  len(train_loader)),
                  "Valid Loss: {:.3f}   |"   .format(valid_loss     /  len(val_loader)),
                  "Train meanIoU:{:.3f}   |" .format(IoU_score      /  len(train_loader)),
                  "Valid meanIoU: {:.3f}   |".format(valid_IoU_score/  len(val_loader)),
                  "Train Accuracy:{:.3f}   |".format(accuracy       /  len(train_loader)),
                  "Valid Accuracy:{:.3f}   |".format(valid_accuracy /  len(val_loader)),
                  "Time: {:.2f}m"            .format((time.time() - since) / 60))
        
        
    history = {'train_loss'       : train_losses,
               'valid_loss'       : valid_losses,
               'train_meanIoU'    : train_IoUs,
               'valid_meanIoU'    : valid_IoUs,
               'train_accuracies' : train_accuracies,
               'valid_accuracies' : valid_accuracies,
               'learn_rates'      : learn_rates}
    
    print('\n \n Training finished! \n Total time: {:.2f} minutes'.format((time.time() - fit_time) / 60))
    
    return history
