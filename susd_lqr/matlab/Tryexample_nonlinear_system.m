clc;
clear all;
close all;
%% The system
%--------------------------------------------------------------------------%
%x1dot=4x2
%x2dot=-x1^3-4x2+u


%--------------------------------------------------------------------------%
%%  The cost parameters
Q = [1 0;  % Penalize angular error
     0 1]; % Penalize angular rate
R = 1;     % Penalize thruster effort
%--------------------------------------------------------------------------%
%% The learning parameters
Number_of_simulated_trajectories=3;
dt=0.01;tf_search=5;tf_forward=7;tf=5;
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
[K_Learned, K_hat,T_consumed]=SUSD_search_nonlinear(Q,R,x(:,1),K_hat_0,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);
linearization_range=0.1;
%[K_Learned, K_hat]=SUSD_search(A,B,C,D,Q,R,x(:,1),K_hat_0,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);  
for i=2:length(t)
                u=-K_Learned*x(:,i-1);
                x1=x(1,i-1);x2=x(2,i-1);
                x1dot=4*x2;
                x2dot=-x1^3-4*x2+u;
                xdot=[x1dot x2dot]';
x(:,i)=x(:,i-1)+dt*xdot;
if norm(x(:,i)-x(:,i-1))>linearization_range
[K_Learned, K_hat,T_consumed]=SUSD_search_nonlinear(Q,R,x(:,i-1),K_hat,dt,tf_search,tf_forward,Number_of_simulated_trajectories,window,accuracy);  
end
end
plot(x(1,:),x(2,:),'LineWidth' ,1.5)
hold on
%% The feedback linearization
%let u=x1^3-x1+v
%The resulting system is
A=[0 4;-1 -4];
B=[0 1]';
%Suppose also
C = [1 0];
D = 0;
%The LQR
[K, P, Poles] = lqr(A,B,Q,R);
sys = ss((A - B*K), B, C, D);
[y,t,x] = initial(sys, x0, t);
plot(x(:,1),x(:,2),'LineWidth' ,1.5)
% %The system is 
% x=zeros(2,length(t));
% x(:,1)=x0;
% for i=2:length(t)
%     x1=x(1,i-1);x2=x(2,i-1);
%     u=x1^3-x1;
%     x1dot=4*x2;
%     x2dot=-x1^3-4*x2+u;
%     xdot=[x1dot x2dot]';
% x(:,i)=x(:,i-1)+dt*xdot;
% end
% plot(x(1,:),x(2,:),'LineWidth' ,1.5)
legend('SUSD','LQR on the Feedback Linearized')
xlabel('x1')
ylabel('x2')
title(sprintf('state trajectory with %d agents',Number_of_simulated_trajectories))
hold off


