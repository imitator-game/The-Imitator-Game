"""CPU tests for absolute-step cosine decay and checkpoint continuation."""
import ast,math,unittest
from pathlib import Path
from types import SimpleNamespace
import torch

path=Path(__file__).resolve().parents[1]/'zero_wam/training.py'
if not path.exists():path=Path(__file__).resolve().parent/'after/training.py'
tree=ast.parse(path.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['cosine_lr_multiplier','build_lr_scheduler']]
space={'math':math,'torch':torch};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),space)
class Config(SimpleNamespace):
 def get(self,key,default=None):return getattr(self,key,default)
config=Config(ig_lr_schedule='cosine',learning_rate=1e-5,ig_min_learning_rate=1e-6,ig_lr_decay_start_step=2500,ig_lr_decay_steps=500)
class TestCosineLR(unittest.TestCase):
 def test_endpoints_and_midpoint(self):
  f=space['cosine_lr_multiplier']
  for step,expected in [(0,1.),(2499,1.),(2500,1.),(2750,.55),(3000,.1),(3500,.1)]:self.assertAlmostEqual(f(step,config),expected)
 def test_monotonic(self):
  vals=[space['cosine_lr_multiplier'](s,config)for s in range(2500,3501)]
  self.assertTrue(all(a>=b for a,b in zip(vals,vals[1:])))
 def test_midpoint_resume_keeps_remaining_half(self):
  parameter=torch.nn.Parameter(torch.tensor([0.]));optimizer=torch.optim.AdamW([parameter],lr=1e-5)
  scheduler=space['build_lr_scheduler'](optimizer,config)
  saved={'base_lrs':[1e-5],'last_epoch':2500,'_step_count':2501,'_is_initial':False,'_get_lr_called_within_step':False,'_last_lr':[1e-5],'lr_lambdas':[None]}
  scheduler.load_state_dict(saved);first=[]
  for step in range(2501,3501):
   optimizer.step();scheduler.step();first.append(scheduler.get_last_lr()[0])
   if step==2750:opt_state=optimizer.state_dict();sched_state=scheduler.state_dict()
  p2=torch.nn.Parameter(torch.tensor([0.]));o2=torch.optim.AdamW([p2],lr=1e-5);s2=space['build_lr_scheduler'](o2,config)
  o2.load_state_dict(opt_state);s2.load_state_dict(sched_state)
  self.assertAlmostEqual(o2.param_groups[0]['lr'],5.5e-6)
  second=[]
  for _ in range(750):o2.step();s2.step();second.append(s2.get_last_lr()[0])
  self.assertEqual(first[250:],second);self.assertAlmostEqual(second[-1],1e-6)

class TestFreshCosineWarmup(unittest.TestCase):
 def test_official_warmup_and_cosine_endpoints(self):
  c=Config(ig_lr_schedule='cosine',learning_rate=1e-4,ig_min_learning_rate=1e-5,ig_lr_decay_start_step=10,ig_lr_decay_steps=490,warmup_steps=10)
  f=space['cosine_lr_multiplier']
  for step,expected in [(0,0.),(1,.1),(5,.5),(10,1.),(255,.55),(500,.1)]:self.assertAlmostEqual(f(step,c),expected)
  p=torch.nn.Parameter(torch.zeros(1));o=torch.optim.AdamW([p],lr=c.learning_rate);s=space['build_lr_scheduler'](o,c)
  self.assertEqual(o.param_groups[0]['lr'],0.)
  rates=[]
  for step in range(1,501):
   o.step();s.step();rates.append(s.get_last_lr()[0])
  self.assertAlmostEqual(rates[9],1e-4);self.assertAlmostEqual(rates[-1],1e-5)
  self.assertTrue(all(a>=b for a,b in zip(rates[9:],rates[10:])))
 def test_existing_continuation_unchanged(self):
  c=Config(learning_rate=1e-5,ig_min_learning_rate=1e-6,ig_lr_decay_start_step=2500,ig_lr_decay_steps=500,warmup_steps=10)
  for step in range(2500,3001):
   expected=.1+.9*.5*(1+math.cos(math.pi*(step-2500)/500))
   self.assertAlmostEqual(space['cosine_lr_multiplier'](step,c),expected)

if __name__=='__main__':unittest.main()
