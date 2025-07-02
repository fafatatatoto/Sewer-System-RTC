# Sewer-System-RTC

Real-time control (RTC) in urban sewer system using rule-based control and deep reinforcement learning method.

## **Purpose**
Due to the corroded and aging sewage pipelines, rainfall invasion to the urban sewer system seems inevitable. Therefore, it is likely that sewer overflows from manholes if there is not appropriate control measures. Besides, with the burgeoning development of Internet of Things (IoT), urban sewer system can be faciliated with sensors and activators connected through the internet so that remote and automated control is becoming increasingly feasible now. Thus, this repo aims to leverage IoT technology in sewer system and develop two types of models:
- Rule-based model (RBC): leveraging predefined control rules for decision making.
- Reinforcement learning model (DRL): leveraging SAC algorithm to provide dynamic suggestions.

  
The goal is to provide operational suggestions for sewer systems and notch up decision-making to mitigate overflow risks.

## **Feature**
- Comparison between rule-based control and reinforcement learning control
- Reinforcement learning model needs to learn a policy that not only optimizes for ultimate objective but also comply with the on-site facility limitations (pumps, sumps, etc).
- Reward engineering: design the reward function for each objective and tune the weights among goals in order to ensure the effectiveness of agent's policy and optimize its performance.

### **Action space design**
 - Since the ttried both discrete and continuous action space, 

### **Reward design**
The objectives includes water level objective, pump operation objective, gate operation objective and power cost objective. Among them, water level objective is the main goal which is intended to maintain the water level of the sump in the pump station in a fixed range, the pump and gate operation objecctive are meant to evaluate the compliance of rules of the on-site facilities, and the power objective is to assess and reduce the power cost of operating pumps.  
- Water level objective: once the water level exceeds the acceptable range, agent will suffer from a large negative reward.
- Pump and gate operation objective: the on-stie facilities have their limitations (e.g. not able to switch on/off frequently), but sometime it's unavoidable to breach the rules (e.g. the coming of surge/ rainwater). Thus, this part uses the frequency of switching on/off to calculate the punishment (if the agent violate the rules more frequent then the penalty would be greater) which enables agent to infringe the rules for greater good but not too often.
- Power cost objective: due to the variability of power price, the reward calculation is designed to be in the power of a coefficient proportional to electricity price. 


## **Results**
- Both rule-based and reinforcement leanrning methods are able to achieve the main goal (maintianing water level), and rarely to break the facility operation rules. 
- In the case of input uncertainty, reinforcement learning demonstrates the superior adaptability and robustness with smaller range of the prediction interval
- With proper reward engineering, even though some objectives are contradicted with each other, RL can learn the trade-off among them, and choose the suitable measures to adjust if necessary. 
<img src="plot/wet_day/20230630_0550_plot_o0.png" alt="not shown" width="410" height="380"/>
  
