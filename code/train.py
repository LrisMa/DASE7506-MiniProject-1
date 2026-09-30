"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from plot_validation import write_validation_svg

# train this model == 训练模型预测下一个token的能力
# 但其实这里只是做一些小的参数修改，例如batch window size，学习率等，来提升模型的性能。
# 控制模型真正性能的code不在这里
def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student') #调用student.py中的StudentModel类来创建模型   
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--min-lr-fraction', type=float, default=.1)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--label-smoothing', type=float, default=0.)
    p.add_argument('--init-checkpoint', type=Path,
                   help='Load model weights from a compatible checkpoint before training.')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Save a model-only checkpoint every N steps; 0 disables snapshots.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='Optional exponential moving average of model weights; 0 disables it.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.warmup_steps < 0:
        p.error('Batch size and step count must be positive; warmup steps must be non-negative.')
    if not 0. <= args.ema_decay < 1.:
        p.error('EMA decay must be in [0, 1).')
    if args.lr <= 0. or not 0. <= args.min_lr_fraction <= 1.:
        p.error('Learning rate must be positive and its minimum fraction must be in [0, 1].')
    if not 0. <= args.beta1 < 1. or not 0. <= args.beta2 < 1.:
        p.error('AdamW betas must be in [0, 1).')
    if not 0. <= args.label_smoothing < 1.:
        p.error('Label smoothing must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(initial['model'])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    ema_parameters = None
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    validation_path = args.run_dir/'validation_history.jsonl'
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        warmup = min(1., (step+1) / max(1, args.warmup_steps)) if args.warmup_steps else 1.
        cosine = args.min_lr_fraction + (1.-args.min_lr_fraction) * .5 * (1.+math.cos(math.pi*step/args.steps))
        learning_rate = args.lr * warmup * cosine
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(), batch[:,1:].flatten(),
                                   label_smoothing=args.label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if args.ema_decay > 0. and step + 1 >= 100:
            with torch.no_grad():
                if ema_parameters is None:
                    ema_parameters = {name: parameter.detach().clone()
                                      for name, parameter in model.named_parameters()}
                else:
                    for name, parameter in model.named_parameters():
                        ema_parameters[name].mul_(args.ema_decay).add_(parameter.detach(), alpha=1.-args.ema_decay)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            with validation_path.open('a') as handle:
                handle.write(json.dumps(validation_history[-1])+'\n')
            write_validation_svg(validation_history, args.run_dir/'validation_curve.svg')
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
        if args.checkpoint_every > 0 and (step+1) % args.checkpoint_every == 0:
            snapshot = args.run_dir/f'checkpoint_step_{step+1}.pt'
            torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                        'model':model.cpu().state_dict(),'seed':args.seed,
                        'train_tokens':(step+1)*args.batch_size*256}, snapshot)
            model.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    raw_validation = None
    if ema_parameters is not None:
        raw_validation = validation
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(ema_parameters[name])
        validation = score(model,*data['validation'],device,'fp32')
        validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,'ema_decay':args.ema_decay},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'raw_validation':raw_validation,'ema_decay':args.ema_decay,
              'lr':args.lr,'min_lr_fraction':args.min_lr_fraction,'warmup_steps':args.warmup_steps,
              'weight_decay':args.weight_decay,'betas':[args.beta1,args.beta2],
              'label_smoothing':args.label_smoothing,'init_checkpoint':str(args.init_checkpoint) if args.init_checkpoint else None,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
