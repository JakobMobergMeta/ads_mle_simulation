
@simulation_bot "
Look at the provided knowldege-notebook for how to use the api
Answer q 1-10 below

q1. How many parameters does a   network with 
 input_dim = 256
   output_dim = 6
   arch = [256, 256, 128]
 architecture have? Answer in int (single digits) 

q2.What's the cost of training a model for 15 days with 16 GPUs per day?      

q3.How much budget remains when no training has been done yet?      

q4.What is the the 10day curve (as a string) for a [512, 512] arch with default input/output dimensions, Asume no-noise. Round to 0.0001.         

q5.What's the train_ne  of an architecture [256, 256] trained for 20 days? Round to 0.0001. Assume no noise and no budget constraints. Assume default input/outut dims               

q6.What's the train_ne difference between 5 days vs 50 days of training (1M params)? Answer: ne_training_5_day - ne_training_50_day. No noise, default input/output dimensions.  Round to 0.0001            

q7.Does noise affect model performance? Test with 1.5M param model, 25 days training, override_noise=0.05. and compare eval_ne. Answer in with true/false

q8.What's the QPS of  30 day training for a 2M param model? Round to single digits. default input/output dimensions, assume no noise

q9.What msg do you get you try to exceed budget (training 2000 days)?

q10.Does model performance improve beyond saturation point? Compare  [2048, 2048]  vs   [2048, 2048,2048, 2048 ], params at 40 days training. Answer as diff ne_largest- ne_smaller , Asume no noise and Round to 0.0001
Respond in a well formatted output

"
