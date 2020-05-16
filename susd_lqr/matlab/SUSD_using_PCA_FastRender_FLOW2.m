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
% Control Law
Q = [1 0;  % Penalize angular error
     0 1]; % Penalize angular rate
R = 1;     % Penalize thruster effort

x0 = [3;  % 3 radians
      0]; % 0 rad/s
%--------------------------------------------------------------------------%
%% Generate Field in the Space of K
%--------------------------------------------------------------------------%
dt=0.01;T=5;t_learning = 0:dt:T;

k1=0:0.5:10;
k2=0:0.5:10;

f=zeros(length(k1),length(k2));
for i=1:length(k1)
    for j=1:length(k2)
        K=[k1(i) k2(j)];
        sys = ss((A - B*K), B, C, D);
        [~,~,x] = initial(sys, x0, t_learning);
        u=-K*x';
        cost=zeros(1,length(u));
        for a=1:length(u)
            cost(1,a)=x(a,:)*Q*x(a,:)'+u(1,a)*R*u(1,a);
        end
        f(i,j)=cost*ones(length(u),1);
        f(i,j)=log(f(i,j));
        
    end
end
[Cmatrix, h] = contour(k1,k2,f',100);
[contour_x, contour_y, contour_levels] = points_from_contourmatrix(Cmatrix);

%--------------------------------------------------------------------------%

% profile on

bound=[ 0 10 0 10]; %Field area
dimension=2;%Select dimension
Number_of_simulated_trajectories=8; %Number of agents

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K0=[-2 -1];
r=0.1;
K_hat=zeros(Number_of_simulated_trajectories,2);
for a=1:Number_of_simulated_trajectories
    U=[rand rand];
    U=r*U/norm(U);
    K_hat(a,:)=K0+U+3;%Initilize the simulated gains
end


%% Generate Graphs
Graph=ones(Number_of_simulated_trajectories,Number_of_simulated_trajectories)-diag(ones(Number_of_simulated_trajectories,1)); %Complete graph
%Graph_L=diag(ones(Number_of_simulated_trajectories-1,1),1)+diag(ones(Number_of_simulated_trajectories-1,1),-1); %Line graph


dt=0.01;  tf=5;
dtau=0.01;eps=0.01;
tau=dt/eps;
k_n_susd=5;% The SUSD gain
k_q_susd=0;
Connectivity_Distance=0.5;%Desired separtaion distance
k_formation=0; %formation in all directions, put it 0 to cancel formation

%  vidObj = VideoWriter('sim_with_terminataion_X');
%  vidObj.FrameRate=20;
% open(vidObj);
fighandle = figure(1);
clear drawing_handles
%% Initial Formation
if k_formation~=0
t_formation=1;
tff=0;
while tff<t_formation
for i=1:Number_of_simulated_trajectories
        Formation=zeros(Number_of_simulated_trajectories,dimension); %formation force in all directions
        I=find(Graph(i,:)); %Find indices of my neighbors
        for j=1:length(I)
            d_ij=norm(K_hat(I(j),:)-K_hat(i,:));
            Formation(i,:)=Formation(i,:)+(d_ij-Connectivity_Distance)*(K_hat(I(j),:)-K_hat(i,:));
        end
        K_hat_dot=k_formation*Formation(i,:);% Update velocities
        K_hat(i,:)=K_hat(i,:)+0.01*K_hat_dot;% Update positions 
end
tff=tff+0.01;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Data = K_hat;
Umean = mean(Data);
R_u = Data - Umean(ones(size(Data,1),1),:);
Covariance = R_u' * R_u;
[Eigenvectors ,Eigenvalues]=eig(Covariance);
n=Eigenvectors(:,1)';q=Eigenvectors(:,2)';
%n=Eigenvalues(1,1)*n/Covariance;

trail.x = K_hat(:,1)';
trail.y = K_hat(:,2)';
trail_idx = 1;
t=0:dt:tf;
Heading=zeros(Number_of_simulated_trajectories,2);
z_center=zeros(1,length(t));
z_min=zeros(1,length(t));
K_learned=zeros(length(t),2);
window=20;
accuracy=30;
for step =1:length(t)
     
        z=zeros(1,Number_of_simulated_trajectories);
        %% calculate cost
    for i=1:Number_of_simulated_trajectories
        sys = ss((A - B*K_hat(i,:)), B, C, D);
        [~,~,x] = initial(sys, x0, t_learning);
        u=-K_hat(i,:)*x';
        cost=zeros(1,length(u));
        for a=1:length(u)
            cost(1,a)=x(a,:)*Q*x(a,:)'+u(1,a)*R*u(1,a);
        end
        z(1,i)=cost*ones(length(u),1);
        %z(1,i)=log(z(1,i));    
    end
    z_center(1,step)=(1/Number_of_simulated_trajectories)*z*ones(Number_of_simulated_trajectories,1);
    K_learned(step,:)=(1/Number_of_simulated_trajectories)*ones(1,Number_of_simulated_trajectories)*K_hat;

  

        %%  Compute formation speed
       for i=1:Number_of_simulated_trajectories
        Formation=zeros(Number_of_simulated_trajectories,dimension); %formation force in all directions
        I=find(Graph(i,:)); %Find indices of my neighbors
        for j=1:length(I)
            d_ij=norm(K_hat(I(j),:)-K_hat(i,:));
            Formation(i,:)=Formation(i,:)+(d_ij-Connectivity_Distance)*(K_hat(I(j),:)-K_hat(i,:));
        end
       end
       %%Update Gains
       zmin=min(z);zmax=max(z);
       z_min(1,step)=zmin;
       minzindex=find(z==zmin);
       maxzindex=find(z==zmax);
       z=1-exp(-(z-zmin));
       if step>window
           if abs(z_min(1,step)-(1/window)*z_min(1,step-window:step-1)*ones(window,1))<accuracy
               break
           end
       end
       for i=1:Number_of_simulated_trajectories

        K_hat_dot=k_n_susd*z(1,i)*n+k_formation*Formation(i,:);% Update velocities
        K_hat(i,:)=K_hat(i,:)+dt*K_hat_dot;% Update positions  
       end
        Data = K_hat;
        Umean = mean(Data);
        R_u = Data - Umean(ones(size(Data,1),1),:);
        Covariance = R_u' * R_u;
        [Eigenvectors ,Eigenvalues]=eig(Covariance);
        nold=n;
        n=Eigenvectors(:,1)';q=Eigenvectors(:,2)';
        if n*nold'<0
            n=-n;q=-q;%to make sure there is no 180 degrees switching.
        end
    trail_idx = trail_idx + 1;
    trail.x(trail_idx,:) = K_hat(:,1)';
    trail.y(trail_idx,:) = K_hat(:,2)';
    
    
        if ~exist('drawing_handles')
            drawing_handles = show_graph_2D_FastRender(step,fighandle,K_hat,bound,Number_of_simulated_trajectories,Graph,contour_x,contour_y,contour_levels, struct(),trail);
        else
            drawing_handles = show_graph_2D_FastRender(step,fighandle,K_hat,bound,Number_of_simulated_trajectories,Graph,contour_x,contour_y,contour_levels, drawing_handles,trail);
        end
        
  
    %pause(1);
%      drawnow limitrate;
%          F = getframe(figure(1));
%        writeVideo(vidObj,F);
%     t=t+dt;
%     step = step + 1;
   
end
%close(vidObj)
index_of_min=find(z==0);
K_Learned=K_hat(index_of_min,:)
[KK, P, Poles] = lqr(A,B,Q,R);
K_Optimal=KK
hold on
plot(K_Learned(1,1),K_Learned(1,2),'d','color','b','MarkerSize',8,'MarkerFaceColor','b')
hold on
plot(K_Optimal(1,1),K_Optimal(1,2),'s','color','k','MarkerSize',8,'MarkerFaceColor','k')

hold off

figure
plot(t(1:step),z_min(1:step),'r','LineWidth' ,1.5)
xlabel('time')
ylabel('z_{min}')
title('Trajectory of the encountered minimum cost ')
  
 
% profile viewer