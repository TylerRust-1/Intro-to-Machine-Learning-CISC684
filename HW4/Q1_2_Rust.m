theta = [1;1];
A=[2,0;0,3];
1*(.5*transpose(theta)*A*theta)
for i =1:10
   theta=theta-1*(.5*transpose(theta)*A*theta);
   theta
end