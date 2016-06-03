
%******************** HOMEWORK 2 ARTIFICIAL INTELLIGENCE ********************%
% by Koyya, Shiva Karthik Reddy


% baysian network 
N=5;
dag=zeros(N,N);

% Assigning node numbers

A=1;B=2;C=3;D=4;E=5;
dag(A,[C D])=1;
dag([A B],D)=1;
dag([C D],E)=1;
discrete_node=1:N;
node_sizes=[2 2 2 2 2];

% making bayes network

bnet=mk_bnet(dag,node_sizes); % BY DEFAULT ALL NOSES ARE DISCRETE 
G=bnet.dag;
figure, draw_graph(G);
title('original network')

% assigning values to the CPT table

bnet.CPD{A}=tabular_CPD(bnet,A,[0.30 0.70]);
bnet.CPD{B}=tabular_CPD(bnet,B,[0.55 0.45]);
bnet.CPD{C}=tabular_CPD(bnet,C,[0.9 0.7 0.1 0.3 ]);
bnet.CPD{D}=tabular_CPD(bnet,D,[0.6 0.5 0.8 0.3 0.4 0.5 0.2 0.7]);
bnet.CPD{E}=tabular_CPD(bnet,E,[0.7 0.1 0.6 0.9 0.3 0.9 0.4 0.1]);


% creating evidence for computing joint probability for A=a1 B=b1 C=c0 D=d0
% E=e1
% false=1 , true=2
evidence=cell(1,N);
evidence{D}=2;
evidence{B}=2;
evidence{C}=1;
evidence{E}=1;
engine=jtree_inf_engine(bnet);
[engine,ll]=enter_evidence(engine,evidence);
m=marginal_nodes(engine,A );
M=m.T;

disp('conditional probabiltity "p=(A/B=2,C=1,D=2,E=1)" with CPD defined by me =')
disp(M)





%% creating data samples from bayesian data

data_sample_50=cell(5,50);
data_sample_500=cell(5,500);
data_sample_2500=cell(5,2500);
data_sample_5000=cell(5,5000);
data_sample_10000=cell(5,10000);

for i=1:50
    data_sample_50(:,i)=sample_bnet(bnet); 
end

data_50=cell2num(data_sample_50);
for i=1:500
    data_sample_500(:,i)=sample_bnet(bnet); 
end
data_500=cell2num(data_sample_500);
for i=1:2500
    data_sample_2500(:,i)=sample_bnet(bnet);
end
data_2500=cell2num(data_sample_2500);
for i=1:5000
    data_sample_5000(:,i)=sample_bnet(bnet);
end
data_5000=cell2num(data_sample_5000);
for i=1:10000
    data_sample_10000(:,i)=sample_bnet(bnet);
end
data_10000=cell2num(data_sample_10000);

%% learning  baseyain network  for different sampled data

bnet1=learn_params(bnet,data_50);
bnet2=learn_params(bnet,data_500);
bnet3=learn_params(bnet,data_2500);
bnet4=learn_params(bnet,data_5000);
bnet5=learn_params(bnet,data_10000);

% joint probability for 50 samples 

engine1=jtree_inf_engine(bnet1);
[engine1,ll]=enter_evidence(engine1,evidence);
m1=marginal_nodes(engine1,A);
M1=m1.T;
disp('CONDITOPNAL probability "p=(A/B=2,C=1,D=2,E=1)" with sampled data for 50 samples ')
disp(M1)

% joint probability for 500 samples 

engine2=jtree_inf_engine(bnet2);
[engine2,ll]=enter_evidence(engine2,evidence);
m2=marginal_nodes(engine2,A);
M2=m2.T;
disp('joint probability "p=(A/B=2,C=1,D=2,E=1)" with sampled data for 500 samples ')
disp(M2)

% joint probability for 2500 samples 

engine3=jtree_inf_engine(bnet3);
[engine3,ll]=enter_evidence(engine3,evidence);
m3=marginal_nodes(engine3,A);
M3=m3.T;
disp('joint probability "p=(A/B=2,C=1,D=2,E=1)" with sampled data for 2500 samples ')
disp(M3)

% joint probability for 5000 samples 

engine4=jtree_inf_engine(bnet4);
[engine4,ll]=enter_evidence(engine4,evidence);
m4=marginal_nodes(engine4,A);
M4=m4.T;
disp('joint probability "p=(A/B=2,C=1,D=2,E=1)" with sampled data for 5000 samples ')
disp(M4)

% joint probability for 10000 samples 

engine5=jtree_inf_engine(bnet5);
[engine5,ll]=enter_evidence(engine5,evidence);
m5=marginal_nodes(engine5,A);
M5=m5.T;
disp('joint probability "p=(A/B=2,C=1,D=2,E=1)" with sampled data for 10000 samples ')
disp(M5)

%% structure learning
% Now we can use the data to learn the network structure and parameters
% We can use a Markov Chain Monte Carlo (MCMC) algorithm to learn the
% structure for diffrent sample sizeof 10000
%% for  with diffrent algorithm paramets

% experimenting with 'nsamples' considered 

[sampled_graphs, accept_ratio] = learn_struct_mcmc(data_10000, node_sizes, 'nsamples', 200*N, 'burnin', 5*N); 
dag2 = sampled_graphs{200*N}; %checking last sample from MCMC chain
figure, draw_graph(dag2);
title('Learned Network for 10000 sample size with nsamples being 200*5 samples ');

[sampled_graphs, accept_ratio] = learn_struct_mcmc(data_10000, node_sizes, 'nsamples', 400*N, 'burnin', 5*N); 
dag3 = sampled_graphs{400*N}; %checking last sample from MCMC chain
figure, draw_graph(dag3);
title('Learned Network for 10000 sample size with just considering last 400*5 samples ');

% experimenting with 'burnin' factor i.e Number of runs until the chain approaches stationarity

[sampled_graphs, accept_ratio] = learn_struct_mcmc(data_10000, node_sizes, 'nsamples', 200*N, 'burnin', 1*N); 
dag4 = sampled_graphs{200*N}; %checking last sample from MCMC chain
figure, draw_graph(dag4);
title('Learned Network for 10000 sample size with burnin being 1*5 times ');

[sampled_graphs, accept_ratio] = learn_struct_mcmc(data_10000, node_sizes, 'nsamples', 200*N, 'burnin', 10*N); 
dag5 = sampled_graphs{200*N}; %checking last sample from MCMC chain
figure, draw_graph(dag5);
title('Learned Network for 10000 sample size with burnin being 10*5 times ');
    


