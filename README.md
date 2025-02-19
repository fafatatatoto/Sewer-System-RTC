# Sewer-System-RTC

Real-time control (RTC) in urban sewer system using rule-based control and reinforcement learning method.

## **Purpose**
Due to the corroded and aging sewage pipelines, rainfall invasion to the urban sewer system seems inevitable. Therefore, sewer overflows from manholes are likely without appropriate control measures. With the burgeoning development of Internet of Things (IoT), urban sewer system can now be faciliated with sensors and activators connected through the internet so that remote and automated control is becoming increasingly feasible. Thus, this repo aims to leverage IoT technology in sewer system and develop two types of models:
- Rule-based model: leveraging predefined control rules for decision making.
- Reinforcement learning model: leveraging SAC algorithm to provide dynamic suggestions.
- Reward engineering
  
The goal is to provide operational suggestions for sewer systems and notch up decision-making to mitigate overflow risks.

## **Feature**
- Comparison between rule-based control and reinforcement learning control
- reinforcement learning model needs to learn a policy that not only optimizes for ultimate objective but also defers to the on-site facility limitations (pumps, sumps, etc).

<img src="plot/wet_day/20230630_0550_plot_o0.png" alt="not shown" width="410" height="380"/>
  
