import turtle as t
from turtle import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

def video(seq,data):
    
    t.bgpic('so.gif')
    t.ht()
    t.title("draw trajectories")
    t.screensize(105*10,68*10,'green')
    t.setup(width=0.65,height=0.73)
    t.pensize(10)
    t.speed(5)
    
    if seq not in data.keys():
        print('No this sequence!')
    else:
        defense_x=[0,2,4,6,8,10,12,14,16,18,20]#
        attacking_x=[22,24,26,28,30,32,34,36,38,40,42]#
        ball_x=[44]
        for i in range(len(defense_x)):
            exec('Def'+str(i)+'=Turtle()')
            exec('Def'+str(i)+'.ht()')
            exec('Def'+str(i)+'.pencolor("blue")')
            exec('Def'+str(i)+'.speed(10)')
            
            
        for i in range(len(attacking_x)):
            exec('Att'+str(i)+'=Turtle()')
            exec('Att'+str(i)+'.ht()')
            exec('Att'+str(i)+'.pencolor("red")')
            exec('Att'+str(i)+'.speed(10)')
            
        for i in range(len(ball_x)):
            exec('Ball'+str(i)+'=Turtle()')
            exec('Ball'+str(i)+'.ht()')
            exec('Ball'+str(i)+'.pencolor("yellow")')
            exec('Ball'+str(i)+'.speed(10)')
            
        for frame in range(len(data[seq])):
            for index, i in enumerate(defense_x):
                if frame == 0:
                    exec("Def"+str(index)+".penup()")
                    exec("Def"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
                    exec("Def"+str(index)+".pendown()")
                else:
                    exec("Def"+str(index)+".st()")
                    exec("Def"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
            for index, i in enumerate(attacking_x):
                if frame == 0:
                    exec("Att"+str(index)+".penup()")
                    exec("Att"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
                    exec("Att"+str(index)+".pendown()")
                else:
                    exec("Att"+str(index)+".st()")
                    exec("Att"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
            for index, i in enumerate(ball_x):
                if frame == 0:
                    exec("Ball"+str(index)+".penup()")
                    exec("Ball"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
                    exec("Ball"+str(index)+".pendown()")
                else:
                    exec("Ball"+str(index)+".st()")
                    exec("Ball"+str(index)+".goto(data[seq][frame,i]*10,data[seq][frame,i+1]*10)")
 
            
if __name__ == '__main__':
    print('play video')
    path1=r'SoccerData\\'
    train_data=pickle.load(open(path1+'small_data.pkl', 'rb'), encoding='bytes')
    video(seq=b'sequence_18', data=train_data)