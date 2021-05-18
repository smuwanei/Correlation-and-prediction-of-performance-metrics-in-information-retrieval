#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
n_members=5
metric_depth='_p1000_10'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_10=X_train_p1000_10_topicwise
y_train_10=Y_train_p1000_10_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_10,y_train_10)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_10,y_train_10)
#predictions for trec 2012 follows below
yhat_2012_10=predict_stacked_model(stacked_model,x_test_2012_p1000_10_topicwise)
tau_dnn_ensemble_2012_p1000_10=tau(y_test_2012_p1000_10_topicwise,yhat_2012_10)
spearman_dnn_ensemble_2012_p1000_10=spearman(y_test_2012_p1000_10_topicwise,yhat_2012_10)
mse_dnn_ensemble_2012_p1000_10=numpy.mean((y_test_2012_p1000_10_topicwise-yhat_2012_10)**2)
r_sqr_dnn_ensemble_2012_p1000_10=sklearn.metrics.r2_score(y_test_2012_p1000_10_topicwise,yhat_2012_10)
print('tau_2012_p1000_10=',tau_dnn_ensemble_2012_p1000_10)
print('spearman_2012_p1000_10=',spearman_dnn_ensemble_2012_p1000_10)
print('mse_dnn_ensemble_2012_p1000_10=',mse_dnn_ensemble_2012_p1000_10)
print('r_sqr_dnn_ensemble_2012_p1000_10=',r_sqr_dnn_ensemble_2012_p1000_10)
#predictions for trec 2013 follows below
yhat_2013_10=predict_stacked_model(stacked_model,x_test_2013_p1000_10_topicwise)
tau_dnn_ensemble_2013_p1000_10=tau(y_test_2013_p1000_10_topicwise,yhat_2013_10)
spearman_dnn_ensemble_2013_p1000_10=spearman(y_test_2013_p1000_10_topicwise,yhat_2013_10)
mse_dnn_ensemble_2013_p1000_10=numpy.mean((y_test_2013_p1000_10_topicwise-yhat_2013_10)**2)
r_sqr_dnn_ensemble_2013_p1000_10=sklearn.metrics.r2_score(y_test_2013_p1000_10_topicwise,yhat_2013_10)
print('tau_2013_p1000_10=',tau_dnn_ensemble_2013_p1000_10)
print('spearman_2013_p1000_10=',spearman_dnn_ensemble_2013_p1000_10)
print('mse_dnn_ensemble_2013_p1000_10=',mse_dnn_ensemble_2013_p1000_10)
print('r_sqr_dnn_ensemble_2013_p1000_10=',r_sqr_dnn_ensemble_2013_p1000_10)
#predictions for trec 2014 follows below
yhat_2014_10=predict_stacked_model(stacked_model,x_test_2014_p1000_10_topicwise)
tau_dnn_ensemble_2014_p1000_10=tau(y_test_2014_p1000_10_topicwise,yhat_2014_10)
spearman_dnn_ensemble_2014_p1000_10=spearman(y_test_2014_p1000_10_topicwise,yhat_2014_10)
mse_dnn_ensemble_2014_p1000_10=numpy.mean((y_test_2014_p1000_10_topicwise-yhat_2014_10)**2)
r_sqr_dnn_ensemble_2014_p1000_10=sklearn.metrics.r2_score(y_test_2014_p1000_10_topicwise,yhat_2014_10)
print('tau_2014_p1000_10=',tau_dnn_ensemble_2014_p1000_10)
print('spearman_2014_p1000_10=',spearman_dnn_ensemble_2014_p1000_10)
print('mse_dnn_ensemble_2014_p1000_10=',mse_dnn_ensemble_2014_p1000_10)
print('r_sqr_dnn_ensemble_2014_p1000_10=',r_sqr_dnn_ensemble_2014_p1000_10)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p1000_15'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_15=X_train_p1000_15_topicwise
y_train_15=Y_train_p1000_15_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_15,y_train_15)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_15,y_train_15)
#predictions for trec 2012 follows below
yhat_2012_15=predict_stacked_model(stacked_model,x_test_2012_p1000_15_topicwise)
tau_dnn_ensemble_2012_p1000_15=tau(y_test_2012_p1000_15_topicwise,yhat_2012_15)
spearman_dnn_ensemble_2012_p1000_15=spearman(y_test_2012_p1000_15_topicwise,yhat_2012_15)
mse_dnn_ensemble_2012_p1000_15=numpy.mean((y_test_2012_p1000_15_topicwise-yhat_2012_15)**2)
r_sqr_dnn_ensemble_2012_p1000_15=sklearn.metrics.r2_score(y_test_2012_p1000_15_topicwise,yhat_2012_15)
print('tau_2012_p1000_15=',tau_dnn_ensemble_2012_p1000_15)
print('spearman_2012_p1000_15=',spearman_dnn_ensemble_2012_p1000_15)
print('mse_dnn_ensemble_2012_p1000_15=',mse_dnn_ensemble_2012_p1000_15)
print('r_sqr_dnn_ensemble_2012_p1000_15=',r_sqr_dnn_ensemble_2012_p1000_15)
#predictions for trec 2013 follows below
yhat_2013_15=predict_stacked_model(stacked_model,x_test_2013_p1000_15_topicwise)
tau_dnn_ensemble_2013_p1000_15=tau(y_test_2013_p1000_15_topicwise,yhat_2013_15)
spearman_dnn_ensemble_2013_p1000_15=spearman(y_test_2013_p1000_15_topicwise,yhat_2013_15)
mse_dnn_ensemble_2013_p1000_15=numpy.mean((y_test_2013_p1000_15_topicwise-yhat_2013_15)**2)
r_sqr_dnn_ensemble_2013_p1000_15=sklearn.metrics.r2_score(y_test_2013_p1000_15_topicwise,yhat_2013_15)
print('tau_2013_p1000_15=',tau_dnn_ensemble_2013_p1000_15)
print('spearman_2013_p1000_15=',spearman_dnn_ensemble_2013_p1000_15)
print('mse_dnn_ensemble_2013_p1000_15=',mse_dnn_ensemble_2013_p1000_15)
print('r_sqr_dnn_ensemble_2013_p1000_15=',r_sqr_dnn_ensemble_2013_p1000_15)
#predictions for trec 2014 follows below
yhat_2014_15=predict_stacked_model(stacked_model,x_test_2014_p1000_15_topicwise)
tau_dnn_ensemble_2014_p1000_15=tau(y_test_2014_p1000_15_topicwise,yhat_2014_15)
spearman_dnn_ensemble_2014_p1000_15=spearman(y_test_2014_p1000_15_topicwise,yhat_2014_15)
mse_dnn_ensemble_2014_p1000_15=numpy.mean((y_test_2014_p1000_15_topicwise-yhat_2014_15)**2)
r_sqr_dnn_ensemble_2014_p1000_15=sklearn.metrics.r2_score(y_test_2014_p1000_15_topicwise,yhat_2014_15)
print('tau_2014_p1000_15=',tau_dnn_ensemble_2014_p1000_15)
print('spearman_2014_p1000_15=',spearman_dnn_ensemble_2014_p1000_15)
print('mse_dnn_ensemble_2014_p1000_15=',mse_dnn_ensemble_2014_p1000_15)
print('r_sqr_dnn_ensemble_2014_p1000_15=',r_sqr_dnn_ensemble_2014_p1000_15)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p1000_20'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_20=X_train_p1000_20_topicwise
y_train_20=Y_train_p1000_20_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_20,y_train_20)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_20,y_train_20)
#predictions for trec 2012 follows below
yhat_2012_20=predict_stacked_model(stacked_model,x_test_2012_p1000_20_topicwise)
tau_dnn_ensemble_2012_p1000_20=tau(y_test_2012_p1000_20_topicwise,yhat_2012_20)
spearman_dnn_ensemble_2012_p1000_20=spearman(y_test_2012_p1000_20_topicwise,yhat_2012_20)
mse_dnn_ensemble_2012_p1000_20=numpy.mean((y_test_2012_p1000_20_topicwise-yhat_2012_20)**2)
r_sqr_dnn_ensemble_2012_p1000_20=sklearn.metrics.r2_score(y_test_2012_p1000_20_topicwise,yhat_2012_20)
print('tau_2012_p1000_20=',tau_dnn_ensemble_2012_p1000_20)
print('spearman_2012_p1000_20=',spearman_dnn_ensemble_2012_p1000_20)
print('mse_dnn_ensemble_2012_p1000_20=',mse_dnn_ensemble_2012_p1000_20)
print('r_sqr_dnn_ensemble_2012_p1000_20=',r_sqr_dnn_ensemble_2012_p1000_20)
#predictions for trec 2013 follows below
yhat_2013_20=predict_stacked_model(stacked_model,x_test_2013_p1000_20_topicwise)
tau_dnn_ensemble_2013_p1000_20=tau(y_test_2013_p1000_20_topicwise,yhat_2013_20)
spearman_dnn_ensemble_2013_p1000_20=spearman(y_test_2013_p1000_20_topicwise,yhat_2013_20)
mse_dnn_ensemble_2013_p1000_20=numpy.mean((y_test_2013_p1000_20_topicwise-yhat_2013_20)**2)
r_sqr_dnn_ensemble_2013_p1000_20=sklearn.metrics.r2_score(y_test_2013_p1000_20_topicwise,yhat_2013_20)
print('tau_2013_p1000_20=',tau_dnn_ensemble_2013_p1000_20)
print('spearman_2013_p1000_20=',spearman_dnn_ensemble_2013_p1000_20)
print('mse_dnn_ensemble_2013_p1000_20=',mse_dnn_ensemble_2013_p1000_20)
print('r_sqr_dnn_ensemble_2013_p1000_20=',r_sqr_dnn_ensemble_2013_p1000_20)
#predictions for trec 2014 follows below
yhat_2014_20=predict_stacked_model(stacked_model,x_test_2014_p1000_20_topicwise)
tau_dnn_ensemble_2014_p1000_20=tau(y_test_2014_p1000_20_topicwise,yhat_2014_20)
spearman_dnn_ensemble_2014_p1000_20=spearman(y_test_2014_p1000_20_topicwise,yhat_2014_20)
mse_dnn_ensemble_2014_p1000_20=numpy.mean((y_test_2014_p1000_20_topicwise-yhat_2014_20)**2)
r_sqr_dnn_ensemble_2014_p1000_20=sklearn.metrics.r2_score(y_test_2014_p1000_20_topicwise,yhat_2014_20)
print('tau_2014_p1000_20=',tau_dnn_ensemble_2014_p1000_20)
print('spearman_2014_p1000_20=',spearman_dnn_ensemble_2014_p1000_20)
print('mse_dnn_ensemble_2014_p1000_20=',mse_dnn_ensemble_2014_p1000_20)
print('r_sqr_dnn_ensemble_2014_p1000_20=',r_sqr_dnn_ensemble_2014_p1000_20)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p1000_25'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_25=X_train_p1000_25_topicwise
y_train_25=Y_train_p1000_25_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_25,y_train_25)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_25,y_train_25)
#predictions for trec 2012 follows below
yhat_2012_25=predict_stacked_model(stacked_model,x_test_2012_p1000_25_topicwise)
tau_dnn_ensemble_2012_p1000_25=tau(y_test_2012_p1000_25_topicwise,yhat_2012_25)
spearman_dnn_ensemble_2012_p1000_25=spearman(y_test_2012_p1000_25_topicwise,yhat_2012_25)
mse_dnn_ensemble_2012_p1000_25=numpy.mean((y_test_2012_p1000_25_topicwise-yhat_2012_25)**2)
r_sqr_dnn_ensemble_2012_p1000_25=sklearn.metrics.r2_score(y_test_2012_p1000_25_topicwise,yhat_2012_25)
print('tau_2012_p1000_25=',tau_dnn_ensemble_2012_p1000_25)
print('spearman_2012_p1000_25=',spearman_dnn_ensemble_2012_p1000_25)
print('mse_dnn_ensemble_2012_p1000_25=',mse_dnn_ensemble_2012_p1000_25)
print('r_sqr_dnn_ensemble_2012_p1000_25=',r_sqr_dnn_ensemble_2012_p1000_25)
#predictions for trec 2013 follows below
yhat_2013_25=predict_stacked_model(stacked_model,x_test_2013_p1000_25_topicwise)
tau_dnn_ensemble_2013_p1000_25=tau(y_test_2013_p1000_25_topicwise,yhat_2013_25)
spearman_dnn_ensemble_2013_p1000_25=spearman(y_test_2013_p1000_25_topicwise,yhat_2013_25)
mse_dnn_ensemble_2013_p1000_25=numpy.mean((y_test_2013_p1000_25_topicwise-yhat_2013_25)**2)
r_sqr_dnn_ensemble_2013_p1000_25=sklearn.metrics.r2_score(y_test_2013_p1000_25_topicwise,yhat_2013_25)
print('tau_2013_p1000_25=',tau_dnn_ensemble_2013_p1000_25)
print('spearman_2013_p1000_25=',spearman_dnn_ensemble_2013_p1000_25)
print('mse_dnn_ensemble_2013_p1000_25=',mse_dnn_ensemble_2013_p1000_25)
print('r_sqr_dnn_ensemble_2013_p1000_25=',r_sqr_dnn_ensemble_2013_p1000_25)
#predictions for trec 2014 follows below
yhat_2014_25=predict_stacked_model(stacked_model,x_test_2014_p1000_25_topicwise)
tau_dnn_ensemble_2014_p1000_25=tau(y_test_2014_p1000_25_topicwise,yhat_2014_25)
spearman_dnn_ensemble_2014_p1000_25=spearman(y_test_2014_p1000_25_topicwise,yhat_2014_25)
mse_dnn_ensemble_2014_p1000_25=numpy.mean((y_test_2014_p1000_25_topicwise-yhat_2014_25)**2)
r_sqr_dnn_ensemble_2014_p1000_25=sklearn.metrics.r2_score(y_test_2014_p1000_25_topicwise,yhat_2014_25)
print('tau_2014_p1000_25=',tau_dnn_ensemble_2014_p1000_25)
print('spearman_2014_p1000_25=',spearman_dnn_ensemble_2014_p1000_25)
print('mse_dnn_ensemble_2014_p1000_25=',mse_dnn_ensemble_2014_p1000_25)
print('r_sqr_dnn_ensemble_2014_p1000_25=',r_sqr_dnn_ensemble_2014_p1000_25)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p1000_30'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'
x_train_30=X_train_p1000_30_topicwise
y_train_30=Y_train_p1000_30_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_30,y_train_30)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_30,y_train_30)
#predictions for trec 2012 follows below
yhat_2012_30=predict_stacked_model(stacked_model,x_test_2012_p1000_30_topicwise)
tau_dnn_ensemble_2012_p1000_30=tau(y_test_2012_p1000_30_topicwise,yhat_2012_30)
spearman_dnn_ensemble_2012_p1000_30=spearman(y_test_2012_p1000_30_topicwise,yhat_2012_30)
mse_dnn_ensemble_2012_p1000_30=numpy.mean((y_test_2012_p1000_30_topicwise-yhat_2012_30)**2)
r_sqr_dnn_ensemble_2012_p1000_30=sklearn.metrics.r2_score(y_test_2012_p1000_30_topicwise,yhat_2012_30)
print('tau_2012_p1000_30=',tau_dnn_ensemble_2012_p1000_30)
print('spearman_2012_p1000_30=',spearman_dnn_ensemble_2012_p1000_30)
print('mse_dnn_ensemble_2012_p1000_30=',mse_dnn_ensemble_2012_p1000_30)
print('r_sqr_dnn_ensemble_2012_p1000_30=',r_sqr_dnn_ensemble_2012_p1000_30)
#predictions for trec 2013 follows below
yhat_2013_30=predict_stacked_model(stacked_model,x_test_2013_p1000_30_topicwise)
tau_dnn_ensemble_2013_p1000_30=tau(y_test_2013_p1000_30_topicwise,yhat_2013_30)
spearman_dnn_ensemble_2013_p1000_30=spearman(y_test_2013_p1000_30_topicwise,yhat_2013_30)
mse_dnn_ensemble_2013_p1000_30=numpy.mean((y_test_2013_p1000_30_topicwise-yhat_2013_30)**2)
r_sqr_dnn_ensemble_2013_p1000_30=sklearn.metrics.r2_score(y_test_2013_p1000_30_topicwise,yhat_2013_30)
print('tau_2013_p1000_30=',tau_dnn_ensemble_2013_p1000_30)
print('spearman_2013_p1000_30=',spearman_dnn_ensemble_2013_p1000_30)
print('mse_dnn_ensemble_2013_p1000_30=',mse_dnn_ensemble_2013_p1000_30)
print('r_sqr_dnn_ensemble_2013_p1000_30=',r_sqr_dnn_ensemble_2013_p1000_30)
#predictions for trec 2014 follows below
yhat_2014_30=predict_stacked_model(stacked_model,x_test_2014_p1000_30_topicwise)
tau_dnn_ensemble_2014_p1000_30=tau(y_test_2014_p1000_30_topicwise,yhat_2014_30)
spearman_dnn_ensemble_2014_p1000_30=spearman(y_test_2014_p1000_30_topicwise,yhat_2014_30)
mse_dnn_ensemble_2014_p1000_30=numpy.mean((y_test_2014_p1000_30_topicwise-yhat_2014_30)**2)
r_sqr_dnn_ensemble_2014_p1000_30=sklearn.metrics.r2_score(y_test_2014_p1000_30_topicwise,yhat_2014_30)
print('tau_2014_p1000_30=',tau_dnn_ensemble_2014_p1000_30)
print('spearman_2014_p1000_30=',spearman_dnn_ensemble_2014_p1000_30)
print('mse_dnn_ensemble_2014_p1000_30=',mse_dnn_ensemble_2014_p1000_30)
print('r_sqr_dnn_ensemble_2014_p1000_30=',r_sqr_dnn_ensemble_2014_p1000_30)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p500_10'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'
x_train_10=X_train_p500_10_topicwise
y_train_10=Y_train_p500_10_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_10,y_train_10)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_10,y_train_10)
#predictions for trec 2012 follows below
yhat_2012_10=predict_stacked_model(stacked_model,x_test_2012_p500_10_topicwise)
tau_dnn_ensemble_2012_p500_10=tau(y_test_2012_p500_10_topicwise,yhat_2012_10)
spearman_dnn_ensemble_2012_p500_10=spearman(y_test_2012_p500_10_topicwise,yhat_2012_10)
mse_dnn_ensemble_2012_p500_10=numpy.mean((y_test_2012_p500_10_topicwise-yhat_2012_10)**2)
r_sqr_dnn_ensemble_2012_p500_10=sklearn.metrics.r2_score(y_test_2012_p500_10_topicwise,yhat_2012_10)
print('tau_2012_p500_10=',tau_dnn_ensemble_2012_p500_10)
print('spearman_2012_p500_10=',spearman_dnn_ensemble_2012_p500_10)
print('mse_dnn_ensemble_2012_p500_10=',mse_dnn_ensemble_2012_p500_10)
print('r_sqr_dnn_ensemble_2012_p500_10=',r_sqr_dnn_ensemble_2012_p500_10)
#predictions for trec 2013 follows below
yhat_2013_10=predict_stacked_model(stacked_model,x_test_2013_p500_10_topicwise)
tau_dnn_ensemble_2013_p500_10=tau(y_test_2013_p500_10_topicwise,yhat_2013_10)
spearman_dnn_ensemble_2013_p500_10=spearman(y_test_2013_p500_10_topicwise,yhat_2013_10)
mse_dnn_ensemble_2013_p500_10=numpy.mean((y_test_2013_p500_10_topicwise-yhat_2013_10)**2)
r_sqr_dnn_ensemble_2013_p500_10=sklearn.metrics.r2_score(y_test_2013_p500_10_topicwise,yhat_2013_10)
print('tau_2013_p500_10=',tau_dnn_ensemble_2013_p500_10)
print('spearman_2013_p500_10=',spearman_dnn_ensemble_2013_p500_10)
print('mse_dnn_ensemble_2013_p500_10=',mse_dnn_ensemble_2013_p500_10)
print('r_sqr_dnn_ensemble_2013_p500_10=',r_sqr_dnn_ensemble_2013_p500_10)
#predictions for trec 2014 follows below
yhat_2014_10=predict_stacked_model(stacked_model,x_test_2014_p500_10_topicwise)
tau_dnn_ensemble_2014_p500_10=tau(y_test_2014_p500_10_topicwise,yhat_2014_10)
spearman_dnn_ensemble_2014_p500_10=spearman(y_test_2014_p500_10_topicwise,yhat_2014_10)
mse_dnn_ensemble_2014_p500_10=numpy.mean((y_test_2014_p500_10_topicwise-yhat_2014_10)**2)
r_sqr_dnn_ensemble_2014_p500_10=sklearn.metrics.r2_score(y_test_2014_p500_10_topicwise,yhat_2014_10)
print('tau_2014_p500_10=',tau_dnn_ensemble_2014_p500_10)
print('spearman_2014_p500_10=',spearman_dnn_ensemble_2014_p500_10)
print('mse_dnn_ensemble_2014_p500_10=',mse_dnn_ensemble_2014_p500_10)
print('r_sqr_dnn_ensemble_2014_p500_10=',r_sqr_dnn_ensemble_2014_p500_10)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p500_15'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'
x_train_15=X_train_p500_15_topicwise
y_train_15=Y_train_p500_15_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_15,y_train_15)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_15,y_train_15)
#predictions for trec 2012 follows below
yhat_2012_15=predict_stacked_model(stacked_model,x_test_2012_p500_15_topicwise)
tau_dnn_ensemble_2012_p500_15=tau(y_test_2012_p500_15_topicwise,yhat_2012_15)
spearman_dnn_ensemble_2012_p500_15=spearman(y_test_2012_p500_15_topicwise,yhat_2012_15)
mse_dnn_ensemble_2012_p500_15=numpy.mean((y_test_2012_p500_15_topicwise-yhat_2012_15)**2)
r_sqr_dnn_ensemble_2012_p500_15=sklearn.metrics.r2_score(y_test_2012_p500_15_topicwise,yhat_2012_15)
print('tau_2012_p500_15=',tau_dnn_ensemble_2012_p500_15)
print('spearman_2012_p500_15=',spearman_dnn_ensemble_2012_p500_15)
print('mse_dnn_ensemble_2012_p500_15=',mse_dnn_ensemble_2012_p500_15)
print('r_sqr_dnn_ensemble_2012_p500_15=',r_sqr_dnn_ensemble_2012_p500_15)
#predictions for trec 2013 follows below
yhat_2013_15=predict_stacked_model(stacked_model,x_test_2013_p500_15_topicwise)
tau_dnn_ensemble_2013_p500_15=tau(y_test_2013_p500_15_topicwise,yhat_2013_15)
spearman_dnn_ensemble_2013_p500_15=spearman(y_test_2013_p500_15_topicwise,yhat_2013_15)
mse_dnn_ensemble_2013_p500_15=numpy.mean((y_test_2013_p500_15_topicwise-yhat_2013_15)**2)
r_sqr_dnn_ensemble_2013_p500_15=sklearn.metrics.r2_score(y_test_2013_p500_15_topicwise,yhat_2013_15)
print('tau_2013_p500_15=',tau_dnn_ensemble_2013_p500_15)
print('spearman_2013_p500_15=',spearman_dnn_ensemble_2013_p500_15)
print('mse_dnn_ensemble_2013_p500_15=',mse_dnn_ensemble_2013_p500_15)
print('r_sqr_dnn_ensemble_2013_p500_15=',r_sqr_dnn_ensemble_2013_p500_15)
#predictions for trec 2014 follows below
yhat_2014_15=predict_stacked_model(stacked_model,x_test_2014_p500_15_topicwise)
tau_dnn_ensemble_2014_p500_15=tau(y_test_2014_p500_15_topicwise,yhat_2014_15)
spearman_dnn_ensemble_2014_p500_15=spearman(y_test_2014_p500_15_topicwise,yhat_2014_15)
mse_dnn_ensemble_2014_p500_15=numpy.mean((y_test_2014_p500_15_topicwise-yhat_2014_15)**2)
r_sqr_dnn_ensemble_2014_p500_15=sklearn.metrics.r2_score(y_test_2014_p500_15_topicwise,yhat_2014_15)
print('tau_2014_p500_15=',tau_dnn_ensemble_2014_p500_15)
print('spearman_2014_p500_15=',spearman_dnn_ensemble_2014_p500_15)
print('mse_dnn_ensemble_2014_p500_15=',mse_dnn_ensemble_2014_p500_15)
print('r_sqr_dnn_ensemble_2014_p500_15=',r_sqr_dnn_ensemble_2014_p500_15)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p500_20'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_20=X_train_p500_20_topicwise
y_train_20=Y_train_p500_20_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_20,y_train_20)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_20,y_train_20)
#predictions for trec 2012 follows below
yhat_2012_20=predict_stacked_model(stacked_model,x_test_2012_p500_20_topicwise)
tau_dnn_ensemble_2012_p500_20=tau(y_test_2012_p500_20_topicwise,yhat_2012_20)
spearman_dnn_ensemble_2012_p500_20=spearman(y_test_2012_p500_20_topicwise,yhat_2012_20)
mse_dnn_ensemble_2012_p500_20=numpy.mean((y_test_2012_p500_20_topicwise-yhat_2012_20)**2)
r_sqr_dnn_ensemble_2012_p500_20=sklearn.metrics.r2_score(y_test_2012_p500_20_topicwise,yhat_2012_20)
print('tau_2012_p500_20=',tau_dnn_ensemble_2012_p500_20)
print('spearman_2012_p500_20=',spearman_dnn_ensemble_2012_p500_20)
print('mse_dnn_ensemble_2012_p500_20=',mse_dnn_ensemble_2012_p500_20)
print('r_sqr_dnn_ensemble_2012_p500_20=',r_sqr_dnn_ensemble_2012_p500_20)
#predictions for trec 2013 follows below
yhat_2013_20=predict_stacked_model(stacked_model,x_test_2013_p500_20_topicwise)
tau_dnn_ensemble_2013_p500_20=tau(y_test_2013_p500_20_topicwise,yhat_2013_20)
spearman_dnn_ensemble_2013_p500_20=spearman(y_test_2013_p500_20_topicwise,yhat_2013_20)
mse_dnn_ensemble_2013_p500_20=numpy.mean((y_test_2013_p500_20_topicwise-yhat_2013_20)**2)
r_sqr_dnn_ensemble_2013_p500_20=sklearn.metrics.r2_score(y_test_2013_p500_20_topicwise,yhat_2013_20)
print('tau_2013_p500_20=',tau_dnn_ensemble_2013_p500_20)
print('spearman_2013_p500_20=',spearman_dnn_ensemble_2013_p500_20)
print('mse_dnn_ensemble_2013_p500_20=',mse_dnn_ensemble_2013_p500_20)
print('r_sqr_dnn_ensemble_2013_p500_20=',r_sqr_dnn_ensemble_2013_p500_20)
#predictions for trec 2014 follows below
yhat_2014_20=predict_stacked_model(stacked_model,x_test_2014_p500_20_topicwise)
tau_dnn_ensemble_2014_p500_20=tau(y_test_2014_p500_20_topicwise,yhat_2014_20)
spearman_dnn_ensemble_2014_p500_20=spearman(y_test_2014_p500_20_topicwise,yhat_2014_20)
mse_dnn_ensemble_2014_p500_20=numpy.mean((y_test_2014_p500_20_topicwise-yhat_2014_20)**2)
r_sqr_dnn_ensemble_2014_p500_20=sklearn.metrics.r2_score(y_test_2014_p500_20_topicwise,yhat_2014_20)
print('tau_2014_p500_20=',tau_dnn_ensemble_2014_p500_20)
print('spearman_2014_p500_20=',spearman_dnn_ensemble_2014_p500_20)
print('mse_dnn_ensemble_2014_p500_20=',mse_dnn_ensemble_2014_p500_20)
print('r_sqr_dnn_ensemble_2014_p500_20=',r_sqr_dnn_ensemble_2014_p500_20)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p500_25'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_25=X_train_p500_25_topicwise
y_train_25=Y_train_p500_25_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_25,y_train_25)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_25,y_train_25)
#predictions for trec 2012 follows below
yhat_2012_25=predict_stacked_model(stacked_model,x_test_2012_p500_25_topicwise)
tau_dnn_ensemble_2012_p500_25=tau(y_test_2012_p500_25_topicwise,yhat_2012_25)
spearman_dnn_ensemble_2012_p500_25=spearman(y_test_2012_p500_25_topicwise,yhat_2012_25)
mse_dnn_ensemble_2012_p500_25=numpy.mean((y_test_2012_p500_25_topicwise-yhat_2012_25)**2)
r_sqr_dnn_ensemble_2012_p500_25=sklearn.metrics.r2_score(y_test_2012_p500_25_topicwise,yhat_2012_25)
print('tau_2012_p500_25=',tau_dnn_ensemble_2012_p500_25)
print('spearman_2012_p500_25=',spearman_dnn_ensemble_2012_p500_25)
print('mse_dnn_ensemble_2012_p500_25=',mse_dnn_ensemble_2012_p500_25)
print('r_sqr_dnn_ensemble_2012_p500_25=',r_sqr_dnn_ensemble_2012_p500_25)
#predictions for trec 2013 follows below
yhat_2013_25=predict_stacked_model(stacked_model,x_test_2013_p500_25_topicwise)
tau_dnn_ensemble_2013_p500_25=tau(y_test_2013_p500_25_topicwise,yhat_2013_25)
spearman_dnn_ensemble_2013_p500_25=spearman(y_test_2013_p500_25_topicwise,yhat_2013_25)
mse_dnn_ensemble_2013_p500_25=numpy.mean((y_test_2013_p500_25_topicwise-yhat_2013_25)**2)
r_sqr_dnn_ensemble_2013_p500_25=sklearn.metrics.r2_score(y_test_2013_p500_25_topicwise,yhat_2013_25)
print('tau_2013_p500_25=',tau_dnn_ensemble_2013_p500_25)
print('spearman_2013_p500_25=',spearman_dnn_ensemble_2013_p500_25)
print('mse_dnn_ensemble_2013_p500_25=',mse_dnn_ensemble_2013_p500_25)
print('r_sqr_dnn_ensemble_2013_p500_25=',r_sqr_dnn_ensemble_2013_p500_25)
#predictions for trec 2014 follows below
yhat_2014_25=predict_stacked_model(stacked_model,x_test_2014_p500_25_topicwise)
tau_dnn_ensemble_2014_p500_25=tau(y_test_2014_p500_25_topicwise,yhat_2014_25)
spearman_dnn_ensemble_2014_p500_25=spearman(y_test_2014_p500_25_topicwise,yhat_2014_25)
mse_dnn_ensemble_2014_p500_25=numpy.mean((y_test_2014_p500_25_topicwise-yhat_2014_25)**2)
r_sqr_dnn_ensemble_2014_p500_25=sklearn.metrics.r2_score(y_test_2014_p500_25_topicwise,yhat_2014_25)
print('tau_2014_p500_25=',tau_dnn_ensemble_2014_p500_25)
print('spearman_2014_p500_25=',spearman_dnn_ensemble_2014_p500_25)
print('mse_dnn_ensemble_2014_p500_25=',mse_dnn_ensemble_2014_p500_25)
print('r_sqr_dnn_ensemble_2014_p500_25=',r_sqr_dnn_ensemble_2014_p500_25)
print('*******************************************************************************')

#'C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
#n_members=1
metric_depth='_p500_30'
str_n_members=str(n_members)
str_metric_depth=str(metric_depth)
path_architecture='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\thirdPaper\\'+str_n_members+'model_graph_'+str_metric_depth+'.png'

x_train_30=X_train_p500_30_topicwise
y_train_30=Y_train_p500_30_topicwise
#creating stacked models
create_sub_models(dnn_baseline_model,n_members,metric_depth,x_train_30,y_train_30)
#loading saved models
members=load_all_models(dnn_baseline_model,n_members,metric_depth)
print('Loaded %d models' %len(members))
#define ensemble model
stacked_model=define_stacked_model(members,path_architecture)
#fit stacked model on test set
fit_stacked_model(stacked_model,x_train_30,y_train_30)
#predictions for trec 2012 follows below
yhat_2012_30=predict_stacked_model(stacked_model,x_test_2012_p500_30_topicwise)
tau_dnn_ensemble_2012_p500_30=tau(y_test_2012_p500_30_topicwise,yhat_2012_30)
spearman_dnn_ensemble_2012_p500_30=spearman(y_test_2012_p500_30_topicwise,yhat_2012_30)
mse_dnn_ensemble_2012_p500_30=numpy.mean((y_test_2012_p500_30_topicwise-yhat_2012_30)**2)
r_sqr_dnn_ensemble_2012_p500_30=sklearn.metrics.r2_score(y_test_2012_p500_30_topicwise,yhat_2012_30)
print('tau_2012_p500_30=',tau_dnn_ensemble_2012_p500_30)
print('spearman_2012_p500_30=',spearman_dnn_ensemble_2012_p500_30)
print('mse_dnn_ensemble_2012_p500_30=',mse_dnn_ensemble_2012_p500_30)
print('r_sqr_dnn_ensemble_2012_p500_30=',r_sqr_dnn_ensemble_2012_p500_30)
#predictions for trec 2013 follows below
yhat_2013_30=predict_stacked_model(stacked_model,x_test_2013_p500_30_topicwise)
tau_dnn_ensemble_2013_p500_30=tau(y_test_2013_p500_30_topicwise,yhat_2013_30)
spearman_dnn_ensemble_2013_p500_30=spearman(y_test_2013_p500_30_topicwise,yhat_2013_30)
mse_dnn_ensemble_2013_p500_30=numpy.mean((y_test_2013_p500_30_topicwise-yhat_2013_30)**2)
r_sqr_dnn_ensemble_2013_p500_30=sklearn.metrics.r2_score(y_test_2013_p500_30_topicwise,yhat_2013_30)
print('tau_2013_p500_30=',tau_dnn_ensemble_2013_p500_30)
print('spearman_2013_p500_30=',spearman_dnn_ensemble_2013_p500_30)
print('mse_dnn_ensemble_2013_p500_30=',mse_dnn_ensemble_2013_p500_30)
print('r_sqr_dnn_ensemble_2013_p500_30=',r_sqr_dnn_ensemble_2013_p500_30)
#predictions for trec 2014 follows below
yhat_2014_30=predict_stacked_model(stacked_model,x_test_2014_p500_30_topicwise)
tau_dnn_ensemble_2014_p500_30=tau(y_test_2014_p500_30_topicwise,yhat_2014_30)
spearman_dnn_ensemble_2014_p500_30=spearman(y_test_2014_p500_30_topicwise,yhat_2014_30)
mse_dnn_ensemble_2014_p500_30=numpy.mean((y_test_2014_p500_30_topicwise-yhat_2014_30)**2)
r_sqr_dnn_ensemble_2014_p500_30=sklearn.metrics.r2_score(y_test_2014_p500_30_topicwise,yhat_2014_30)
print('tau_2014_p500_30=',tau_dnn_ensemble_2014_p500_30)
print('spearman_2014_p500_30=',spearman_dnn_ensemble_2014_p500_30)
print('mse_dnn_ensemble_2014_p500_30=',mse_dnn_ensemble_2014_p500_30)
print('r_sqr_dnn_ensemble_2014_p500_30=',r_sqr_dnn_ensemble_2014_p500_30)
print('*******************************************************************************')

