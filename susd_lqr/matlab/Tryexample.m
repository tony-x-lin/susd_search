clc;
clear all;
close all;
%% The system
%--------------------------------------------------------------------------%
A = [1   1; 
     0.01 0];
B = [0; 
     1];
C = [1 0];
D = 0;
%--------------------------------------------------------------------------%
%%  The cost parameters
Q = [1 0;  % Penalize angular error
     0 1]; % Penalize angular rate
R = 1;     % Penalize thruster effort
%--------------------------------------------------------------------------%
%% The learning parameters
Number_of_simulated_trajectories=3;
dt=0.01;tf_search=5;tf_forward=5;tf=5;
window=5;
accuracy=30;
%--------------------------------------------------------------------------%
%% Initialization
x0 = [3;  % 3 radians
      1]; % 0 rad/s
K0=[2 2];
r=0.1;
K_hat=zeros(Number_of_simulated_trajectories,2);
for a=1:Number_of_simulated_trajectories
    U=[rand rand];
    U=r*U/norm(U);
    K_hat(a,:)=K0+U+3;%Initilize the simulated gains
end  
K_hat_0=K_hat;
%--------------------------------------------------------------------------%

t=0:dt:tf;
x=zeros(2,length(t));
x(:,1)=x0;
[K_Learned, K_hat,T_consumed]=SUSD_search(A,B,C,D,Q,R,x(:,1),K_hat_0,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);
linearization_range=0.1;
%[K_Learned, K_hat]=SUSD_search(A,B,C,D,Q,R,x(:,1),K_hat_0,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);  
for i=2:length(t)
   
x(:,i)=x(:,i-1)+dt*(A - B*K_Learned)*x(:,i-1);
if norm(x(:,i)-x(:,i-1))>linearization_range
[K_Learned, K_hat,T_consumed]=SUSD_search(A,B,C,D,Q,R,x(:,i-1),K_hat,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);  
end
end
plot(x(1,:),x(2,:),'LineWidth' ,1.5)
hold on


%% The Exact LQR
[K, P, Poles] = lqr(A,B,Q,R);
sys = ss((A - B*K), B, C, D);
[y,t,x] = initial(sys, x0, t);
plot(x(:,1),x(:,2),'LineWidth' ,1.5)
hold off
legend('SUSD','Exact')
xlabel('x1')
ylabel('x2')
title(sprintf('state trajectory with %d agents',Number_of_simulated_trajectories))

