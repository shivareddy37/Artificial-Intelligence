%%%%%%%%************* Shiva Karthik Reddy Koyya************%%%%%%%%%%%%
%% HW 1 optimization using genetic algorithm and simulated annealing%%

%%
%%%%%%%%%%%%%%%%%% analysing the algorithms %%%%%%%%%%%%%%%%%%%%

%% simulated anneling for levy function
%% the section below also produce plots for part b 

rs_l = RandomStartPointSet('NumStartPoints',2,'ArtificialBound',10); % 2 points
problem = createOptimProblem('fminunc','x0',ones(1,1));
x0_l=list(rs_l,problem);
%x0_l=[-7,12];
lb_l = -5;
ub_l = 5;

LIMS_l=[-20 20];
options_sa1 = saoptimset('PlotFcns',{@saplotf});
[x_sa_levy,fval_sa_levy,exitFlag_sa_levy,output_sa_levy]=simulannealbnd(@levy,x0_l',lb_l,ub_l,options_sa1);

disp('displaying output for levy function')
disp(output_sa_levy)
disp('minimum value for the function within limits')
disp(x_sa_levy)
disp('best value for the function at minimum x') 
disp(fval_sa_levy)
hold on 
fplot(@levy,LIMS_l)
hold on
plot(levy(x_sa_levy),'*r','markersize',10)
hold on
plot(levy(x0_l'),'+c','markersize',10)
title('progression of the annelaing process for levy')
xlabel('Xvalue')
ylabel('function value')
p=legend('progession of algorithm','1-D levy fuction','end point','start point');
set(p,'Location','NorthWest')
hold off

% Genetic algorithm for levy function

options_ga1=gaoptimset('PlotFcns',{ @gaplotbestf});
[x_ga_levy,fval_ga_levy,exitflag_ga_levy,output_ga_levy]=ga(@levy,2,[],[],[],[],lb_l,ub_l,[],options_ga1);
disp('displaying output for levy function with GA optimisation')
disp(output_ga_levy)
disp('minimum value for the function within limits')
disp(x_ga_levy)
disp('best value for the function at minimum x') 
disp(fval_ga_levy)
hold on
fplot(@levy,LIMS_l)
hold on
plot(levy(x0_l'),'+g','markersize',10)
hold on
plot(levy(x_ga_levy),'*r','markersize',10)
hold on
title('progression of the genetic algorithm for levy function')
xlabel('Xvalue')
ylabel('function value')
p=legend('mean fitness','progession of algorithm','1-D levy fuction','start point','end point');
set(p,'Location','NorthWest')
hold off

%% simulated anneling for drop function
rs_d = RandomStartPointSet('NumStartPoints',2,'ArtificialBound',5); % 2 points
problem = createOptimProblem('fminunc','x0',ones(1,1));
x0_d=list(rs_d,problem);
lb_d=-5;
ub_d=5;

LIMS_d=[-5 5];

[x_sa_drop,fval_sa_drop,exitFlag_sa_drop,output_drop]=simulannealbnd(@drop,x0_d',lb_d,ub_d);

disp('displaying output for drop function using simulated anneling')
disp(output_drop)
disp('minimum value for the function within limits')
disp(x_sa_drop)
disp('best value for the function at minimum x') 
disp(fval_sa_drop)

%% Genetic algorithm for drop fumction
[x_ga_drop,fval_ga_drop,exitflag_ga_drop,output_ga_drop]=ga(@levy,2,[],[],[],[],lb_d,ub_d,[]);
disp('displaying output for drop function with GA optimisation')
disp(output_ga_drop)
disp('minimum value for the function within limits')
disp(x_ga_drop)
disp('best value for the function at minimum x') 
disp(fval_ga_drop)

%% part a
%% this part each algorithm is implmented for each function 30 times 
bestfval=zeros(30,4);

for i=1:30
  %levy for levy function
  [x_sa_levy,fval_sa_levy]=simulannealbnd(@levy,x0_l',lb_l,ub_l);
  [x_ga_levy,fval_ga_levy]=ga(@levy,2,[],[],[],[],lb_l,ub_l);
  % for drop fuction
  [x_sa_drop,fval_sa_drop]=simulannealbnd(@drop,x0_d',lb_d,ub_d);
  [x_ga_drop,fval_ga_drop]=ga(@levy,2,[],[],[],[],lb_d,ub_d);
  
  
  bestfval(i,1)=fval_ga_levy;
  bestfval(i,2)=fval_sa_levy;
  bestfval(i,3)=fval_ga_drop;
  bestfval(i,4)=fval_sa_drop;

end
iter=1:30;
figure(5)
plot(iter,bestfval(:,1)','--r')
hold on
plot(min(bestfval(:,1)),'ok','markersize',10)
hold on
plot(mean(bestfval(:,1)),'xm','markersize',10)
plot(iter,bestfval(:,3)')
hold on
plot(min(bestfval(:,3)),'*r','markersize',10)
hold on
plot(mean(bestfval(:,3)),'^g','markersize',10)
title('comparision of both the minimisation algorithms for levy  function')
t=legend('best value for each iteration with simulated anneling','min best value of all iteration for SA','avg of best values for SA'...
    ,'best value for each iteration with genetic algorithm','min best value of all iteration for GA','avg of best values for GA' );
set(t,'Location','NorthEast')
hold off
