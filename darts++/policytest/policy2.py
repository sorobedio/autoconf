import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from model import  CustNetworkCIFAR
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')#0.025
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
parser.add_argument('--fepochs', type=int, default=800, help='num of training final epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
#   model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
#   model = model.cuda()

#   logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()


  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)


  ordering =[]
  depth=13
  nlayers=0
  model1 = CustNetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.auxiliary, genotype, normal=True, step=0)
#   ordering.append(True)
  normal=True
  ng=0
  layers=20
#   # model1 = CustNetworkCIFAR(26, num_classes=10, auxiliary=False, genotype=DARTS_V2, normal=True, step=0)
  for s in range(depth):
    

    step_acc = []
    model1.add_cell(normal=normal)
    model = copy.deepcopy( model1)
    model = model.cuda()
#     model =torch.nn.DataParallel(model).cuda()
    
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best=0.0
    mybest=0
    for epoch in range(args.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)

      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
      if best<valid_acc:
        best=valid_acc

#       utils.save(model, os.path.join(args.save, 'weights1.pt'))
#     step_acc.append(best.item())
    model = model.cpu()
    del model
##################################################################
    model2 =copy.deepcopy( model1)
    if normal==True:
        model2.replace_cell(normal=False)
#         ng +=1
#         normal=False
    else:
        model2.replace_cell(normal=True)
#          normal=True
    model = copy.deepcopy( model2)
    model = model.cuda()
#     model =torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best2=0.0
    for epoch in range(args.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)

      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
    

#       utils.save(model, os.path.join(args.save, 'weights1.pt'))
      if best2<valid_acc:
        best2=valid_acc
#     step_acc.append(best2.item())
#     del model
    model = model.cpu()
    del model
    if best>best2:
            
            mybest=best
          # model=model1.clone()
#             ordering[-1] = normal
            model1.selected_cell(normal)
            ordering.append(normal)
#             ng += normal
#             model1.add_cell(normal=normal)
    #       model1.selected_cell(True)
#             ordering.append(normal)
#             normal =True
    else:
      mybest=best2
      model1 = model2
#       ordering[-1]=not normal
      model1.selected_cell(not normal)
#       model1.add_cell(normal=not normal)
#       ordering.append(not normal)
      normal =not normal
      ordering.append(normal)
    if normal==False:
        ng += 1

    logging.info('depth %d ordering: %s best_acc: %f', s, ordering, mybest)
    if ng==1:
        nlayers= layers-s
        break
    else:
        nlayers=layers-s

#   model1.add_cell(normal=True) 
#   model1.selected_cell(True)
#   ordering.append(True)
#   model1.add_cell(normal=True) 
#   model1.selected_cell(True)
#   ordering.append(True)
#   model1.add_cell(normal=False) 
#   model1.selected_cell(False)
#   ordering.append(False)
#   if 20-nlayers>0:
  for i in range(nlayers):
        if i==13:
            model1.add_cell(normal=False) 
            model1.selected_cell(False)
            ordering.append(False)
        else:
            model1.add_cell(normal=True) 
            model1.selected_cell(True)
            ordering.append(True)
  logging.info('depth 8 ordering: %s ', ordering)
    
    
    
##################################################################TRAIN THE OPTIMAL STACK##########################
    
    #     step_acc = []
  model = copy.deepcopy( model1)
  del model1
  logging.info('-------------Starting with searched architecture training----------------------------')
  model = model.cuda()
    #     model =torch.nn.DataParallel(model).cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
    )

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.fepochs))
  best_acc=0.0
  for epoch in range(args.fepochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.fepochs

      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)

      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
      if best_acc<valid_acc:
        best_acc=valid_acc
        utils.save(model, os.path.join(args.save, 'weights_best.pt'))
      logging.info('best valid_acc %f', best_acc)
    #     step_acc.append(best.item())
    #     mybest=0
    #     args.fepochs=100
   




    ##################################################################TRAIN THE OPTIMAL STACK##########################







def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

