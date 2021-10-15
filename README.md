# S4

Part 1


BackPropagation Details :

input1=i1
input2=i2
actual_output=t1
actual_output=t2
h1=i1*w1+i2*w2
h2=i1*w3+i2*w4
act_h1=sigmoid(h1)
act_h2=sigmoid(h2)
o1= act_h1*w5 + act_h2*w6
o2= act_h1*w7 + act_h2*w8
act_o1=sigmoid(o1)
act_o2=sigmoid(o2)
E1=1/2*( t1 - act_o1)2
E2=1/2*(t2 - act_o2)2
E_total=E1+E2



![Capture](https://user-images.githubusercontent.com/53977148/137501531-0557966e-025d-4c9c-add3-eaf181b147ec.PNG)

