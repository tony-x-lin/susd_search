function [K_Learned, K_hat,T_consumed]=SUSD_search(A,B,C,D,Q,R,x0,K_hat_0,dt,tf,tf_learning,Number_of_simulated_trajectories,window,accuracy)

%% times and gains
%--------------------------------------------------------------------------%
t_learning = 0:dt:tf_learning;
t=0:dt:tf;
K_hat=K_hat_0; 
k_n_susd=5;% The SUSD gain
%--------------------------------------------------------------------------% 
z_min=zeros(1,length(t)); %trajectory of all instantaneous minimum costs
for step =1:length(t)
        %% PCA on K_hat
        %--------------------------------------------------------------------------%
        Data = K_hat;
        Umean = mean(Data);
        R_u = Data - Umean(ones(size(Data,1),1),:);
        Covariance = R_u' * R_u;
        [Eigenvectors ,~]=eig(Covariance);
        n=Eigenvectors(:,1)';
        if step>1 && n*nold'<0
            n=-n;%to make sure there is no 180 degrees switching.
        end
        nold=n;
        %--------------------------------------------------------------------------%

        z=zeros(1,Number_of_simulated_trajectories);
        %% calculate LQR cost by runing the system forward in time
        %--------------------------------------------------------------------------%
        for i=1:Number_of_simulated_trajectories
            sys = ss((A - B*K_hat(i,:)), B, C, D);
            [~,~,x] = initial(sys, x0, t_learning);
            u=-K_hat(i,:)*x';
            cost=zeros(1,length(u));
            for a=1:length(u)
                cost(1,a)=x(a,:)*Q*x(a,:)'+u(1,a)*R*u(1,a);
            end
            z(1,i)=cost*ones(length(u),1);
        end
        %--------------------------------------------------------------------------%
       zmin=min(z); %The instantaneous cost
       z_min(1,step)=zmin; %Save the instantaneous cost
       z=1-exp(-(z-zmin)); %Cost transformation
       %% Stop the search when we reach the desired accuaracy
       %--------------------------------------------------------------------------%
       if step>window
           if abs(z_min(1,step)-(1/window)*z_min(1,step-window:step-1)*ones(window,1))<accuracy
               T_consumed=t(step);
               break
           end
       end
       %--------------------------------------------------------------------------%
       %% Update the SUSD dynamics
       %--------------------------------------------------------------------------%
       for i=1:Number_of_simulated_trajectories
        K_hat_dot=k_n_susd*z(1,i)*n;% Update velocities
        K_hat(i,:)=K_hat(i,:)+dt*K_hat_dot;% Update positions  
       end
       %--------------------------------------------------------------------------% 
end
%% Return the learned cost
index_of_min=find(z==0);
K_Learned=K_hat(index_of_min,:);
end
