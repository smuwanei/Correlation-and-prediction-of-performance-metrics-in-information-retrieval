#setting up training data for high cost precision metric
#replace this path to yours were TREC performance metrics are
path='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\topicwise_dataset_complete_nozeros.csv'
column_names_p1000_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','p1000']
column_names_p1000_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','p1000']
column_names_p1000_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','p1000']
column_names_p1000_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','p1000']
column_names_p1000_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','p1000']
column_names_p1000_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','p1000']
column_names_p1000_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','p1000']
column_names_p1000_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','p1000']
column_names_p1000_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','p1000']
trec_exclude=['WT2012','WT2013','WT2014']
data_p1000_topic_10_training=datasetload_training(column_names_p1000_topic_10,path,trec_exclude)
data_p1000_topic_15_training=datasetload_training(column_names_p1000_topic_15,path,trec_exclude)
data_p1000_topic_20_training=datasetload_training(column_names_p1000_topic_20,path,trec_exclude)
data_p1000_topic_25_training=datasetload_training(column_names_p1000_topic_25,path,trec_exclude)
data_p1000_topic_30_training=datasetload_training(column_names_p1000_topic_30,path,trec_exclude)
data_p1000_topic_35_training=datasetload_training(column_names_p1000_topic_35,path,trec_exclude)
data_p1000_topic_40_training=datasetload_training(column_names_p1000_topic_40,path,trec_exclude)
data_p1000_topic_45_training=datasetload_training(column_names_p1000_topic_45,path,trec_exclude)
data_p1000_topic_50_training=datasetload_training(column_names_p1000_topic_50,path,trec_exclude)
#############################################
column_names_p100_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','p100']
column_names_p100_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','p100']
column_names_p100_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','p100']
column_names_p100_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','p100']
column_names_p100_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','p100']
column_names_p100_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','p100']
column_names_p100_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','p100']
column_names_p100_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','p100']
column_names_p100_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','p100']
trec_exclude=['WT2012','WT2013','WT2014']
data_p100_topic_10_training=datasetload_training(column_names_p100_topic_10,path,trec_exclude)
data_p100_topic_15_training=datasetload_training(column_names_p100_topic_15,path,trec_exclude)
data_p100_topic_20_training=datasetload_training(column_names_p100_topic_20,path,trec_exclude)
data_p100_topic_25_training=datasetload_training(column_names_p100_topic_25,path,trec_exclude)
data_p100_topic_30_training=datasetload_training(column_names_p100_topic_30,path,trec_exclude)
data_p100_topic_35_training=datasetload_training(column_names_p100_topic_35,path,trec_exclude)
data_p100_topic_40_training=datasetload_training(column_names_p100_topic_40,path,trec_exclude)
data_p100_topic_45_training=datasetload_training(column_names_p100_topic_45,path,trec_exclude)
data_p100_topic_50_training=datasetload_training(column_names_p100_topic_50,path,trec_exclude)
###################################################
column_names_p500_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','p500']
column_names_p500_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','p500']
column_names_p500_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','p500']
column_names_p500_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','p500']
column_names_p500_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','p500']
column_names_p500_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','p500']
column_names_p500_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','p500']
column_names_p500_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','p500']
column_names_p500_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','p500']
trec_exclude=['WT2012','WT2013','WT2014']
data_p500_topic_10_training=datasetload_training(column_names_p500_topic_10,path,trec_exclude)
data_p500_topic_15_training=datasetload_training(column_names_p500_topic_15,path,trec_exclude)
data_p500_topic_20_training=datasetload_training(column_names_p500_topic_20,path,trec_exclude)
data_p500_topic_25_training=datasetload_training(column_names_p500_topic_25,path,trec_exclude)
data_p500_topic_30_training=datasetload_training(column_names_p500_topic_30,path,trec_exclude)
data_p500_topic_35_training=datasetload_training(column_names_p500_topic_35,path,trec_exclude)
data_p500_topic_40_training=datasetload_training(column_names_p500_topic_40,path,trec_exclude)
data_p500_topic_45_training=datasetload_training(column_names_p500_topic_45,path,trec_exclude)
data_p500_topic_50_training=datasetload_training(column_names_p500_topic_50,path,trec_exclude)

#setting up training data for high cost ndcg metric
column_names_ndcg1000_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','ndcg1000']
column_names_ndcg1000_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','ndcg1000']
column_names_ndcg1000_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','ndcg1000']
column_names_ndcg1000_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','ndcg1000']
column_names_ndcg1000_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','ndcg1000']
column_names_ndcg1000_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','ndcg1000']
column_names_ndcg1000_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','ndcg1000']
column_names_ndcg1000_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','ndcg1000']
column_names_ndcg1000_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','ndcg1000']
trec_exclude=['WT2012','WT2013','WT2014']
data_ndcg1000_topic_10_training=datasetload_training(column_names_ndcg1000_topic_10,path,trec_exclude)
data_ndcg1000_topic_15_training=datasetload_training(column_names_ndcg1000_topic_15,path,trec_exclude)
data_ndcg1000_topic_20_training=datasetload_training(column_names_ndcg1000_topic_20,path,trec_exclude)
data_ndcg1000_topic_25_training=datasetload_training(column_names_ndcg1000_topic_25,path,trec_exclude)
data_ndcg1000_topic_30_training=datasetload_training(column_names_ndcg1000_topic_30,path,trec_exclude)
data_ndcg1000_topic_35_training=datasetload_training(column_names_ndcg1000_topic_35,path,trec_exclude)
data_ndcg1000_topic_40_training=datasetload_training(column_names_ndcg1000_topic_40,path,trec_exclude)
data_ndcg1000_topic_45_training=datasetload_training(column_names_ndcg1000_topic_45,path,trec_exclude)
data_ndcg1000_topic_50_training=datasetload_training(column_names_ndcg1000_topic_50,path,trec_exclude)
#############################################
column_names_ndcg500_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','ndcg500']
column_names_ndcg500_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','ndcg500']
column_names_ndcg500_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','ndcg500']
column_names_ndcg500_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','ndcg500']
column_names_ndcg500_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','ndcg500']
column_names_ndcg500_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','ndcg500']
column_names_ndcg500_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','ndcg500']
column_names_ndcg500_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','ndcg500']
column_names_ndcg500_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','ndcg500']
trec_exclude=['WT2012','WT2013','WT2014']
data_ndcg500_topic_10_training=datasetload_training(column_names_ndcg500_topic_10,path,trec_exclude)
data_ndcg500_topic_15_training=datasetload_training(column_names_ndcg500_topic_15,path,trec_exclude)
data_ndcg500_topic_20_training=datasetload_training(column_names_ndcg500_topic_20,path,trec_exclude)
data_ndcg500_topic_25_training=datasetload_training(column_names_ndcg500_topic_25,path,trec_exclude)
data_ndcg500_topic_30_training=datasetload_training(column_names_ndcg500_topic_30,path,trec_exclude)
data_ndcg500_topic_35_training=datasetload_training(column_names_ndcg500_topic_35,path,trec_exclude)
data_ndcg500_topic_40_training=datasetload_training(column_names_ndcg500_topic_40,path,trec_exclude)
data_ndcg500_topic_45_training=datasetload_training(column_names_ndcg500_topic_45,path,trec_exclude)
data_ndcg500_topic_50_training=datasetload_training(column_names_ndcg500_topic_50,path,trec_exclude)
###################################################
column_names_ndcg100_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','ndcg100']
column_names_ndcg100_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','ndcg100']
column_names_ndcg100_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','ndcg100']
column_names_ndcg100_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','ndcg100']
column_names_ndcg100_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','ndcg100']
column_names_ndcg100_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','ndcg100']
column_names_ndcg100_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','ndcg100']
column_names_ndcg100_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','ndcg100']
column_names_ndcg100_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','ndcg100']
trec_exclude=['WT2012','WT2013','WT2014']
data_ndcg100_topic_10_training=datasetload_training(column_names_ndcg100_topic_10,path,trec_exclude)
data_ndcg100_topic_15_training=datasetload_training(column_names_ndcg100_topic_15,path,trec_exclude)
data_ndcg100_topic_20_training=datasetload_training(column_names_ndcg100_topic_20,path,trec_exclude)
data_ndcg100_topic_25_training=datasetload_training(column_names_ndcg100_topic_25,path,trec_exclude)
data_ndcg100_topic_30_training=datasetload_training(column_names_ndcg100_topic_30,path,trec_exclude)
data_ndcg100_topic_35_training=datasetload_training(column_names_ndcg100_topic_35,path,trec_exclude)
data_ndcg100_topic_40_training=datasetload_training(column_names_ndcg100_topic_40,path,trec_exclude)
data_ndcg100_topic_45_training=datasetload_training(column_names_ndcg100_topic_45,path,trec_exclude)
data_ndcg100_topic_50_training=datasetload_training(column_names_ndcg100_topic_50,path,trec_exclude)

#setting up training data for high cost rbp metric
column_names_rbp095_1000_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','rbp095_1000']
column_names_rbp095_1000_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','rbp095_1000']
column_names_rbp095_1000_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','rbp095_1000']
column_names_rbp095_1000_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','rbp095_1000']
column_names_rbp095_1000_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','rbp095_1000']
column_names_rbp095_1000_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','rbp095_1000']
column_names_rbp095_1000_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','rbp095_1000']
column_names_rbp095_1000_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','rbp095_1000']
column_names_rbp095_1000_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','rbp095_1000']
trec_exclude=['WT2012','WT2013','WT2014']
data_rbp095_1000_topic_10_training=datasetload_training(column_names_rbp095_1000_topic_10,path,trec_exclude)
data_rbp095_1000_topic_15_training=datasetload_training(column_names_rbp095_1000_topic_15,path,trec_exclude)
data_rbp095_1000_topic_20_training=datasetload_training(column_names_rbp095_1000_topic_20,path,trec_exclude)
data_rbp095_1000_topic_25_training=datasetload_training(column_names_rbp095_1000_topic_25,path,trec_exclude)
data_rbp095_1000_topic_30_training=datasetload_training(column_names_rbp095_1000_topic_30,path,trec_exclude)
data_rbp095_1000_topic_35_training=datasetload_training(column_names_rbp095_1000_topic_35,path,trec_exclude)
data_rbp095_1000_topic_40_training=datasetload_training(column_names_rbp095_1000_topic_40,path,trec_exclude)
data_rbp095_1000_topic_45_training=datasetload_training(column_names_rbp095_1000_topic_45,path,trec_exclude)
data_rbp095_1000_topic_50_training=datasetload_training(column_names_rbp095_1000_topic_50,path,trec_exclude)
#############################################
column_names_rbp095_100_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','rbp095_100']
column_names_rbp095_100_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','rbp095_100']
column_names_rbp095_100_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','rbp095_100']
column_names_rbp095_100_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','rbp095_100']
column_names_rbp095_100_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','rbp095_100']
column_names_rbp095_100_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','rbp095_100']
column_names_rbp095_100_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','rbp095_100']
column_names_rbp095_100_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','rbp095_100']
column_names_rbp095_100_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','rbp095_100']
trec_exclude=['WT2012','WT2013','WT2014']
data_rbp095_100_topic_10_training=datasetload_training(column_names_rbp095_100_topic_10,path,trec_exclude)
data_rbp095_100_topic_15_training=datasetload_training(column_names_rbp095_100_topic_15,path,trec_exclude)
data_rbp095_100_topic_20_training=datasetload_training(column_names_rbp095_100_topic_20,path,trec_exclude)
data_rbp095_100_topic_25_training=datasetload_training(column_names_rbp095_100_topic_25,path,trec_exclude)
data_rbp095_100_topic_30_training=datasetload_training(column_names_rbp095_100_topic_30,path,trec_exclude)
data_rbp095_100_topic_35_training=datasetload_training(column_names_rbp095_100_topic_35,path,trec_exclude)
data_rbp095_100_topic_40_training=datasetload_training(column_names_rbp095_100_topic_40,path,trec_exclude)
data_rbp095_100_topic_45_training=datasetload_training(column_names_rbp095_100_topic_45,path,trec_exclude)
data_rbp095_100_topic_50_training=datasetload_training(column_names_rbp095_100_topic_50,path,trec_exclude)
###################################################
column_names_rbp095_500_topic_10=['p10','ndcg10','bpref10','err10','rbp095_10','infAp10','rbp095_500']
column_names_rbp095_500_topic_15=['p15','ndcg15','bpref15','err15','rbp095_15','infAp15','rbp095_500']
column_names_rbp095_500_topic_20=['p20','ndcg20','bpref20','err20','rbp095_20','infAp20','rbp095_500']
column_names_rbp095_500_topic_25=['p25','ndcg25','bpref25','err25','rbp095_25','infAp25','rbp095_500']
column_names_rbp095_500_topic_30=['p30','ndcg30','bpref30','err30','rbp095_30','infAp30','rbp095_500']
column_names_rbp095_500_topic_35=['p35','ndcg35','bpref35','err35','rbp095_35','infAp35','rbp095_500']
column_names_rbp095_500_topic_40=['p40','ndcg40','bpref40','err40','rbp095_40','infAp40','rbp095_500']
column_names_rbp095_500_topic_45=['p45','ndcg45','bpref45','err45','rbp095_45','infAp45','rbp095_500']
column_names_rbp095_500_topic_50=['p50','ndcg50','bpref50','err50','rbp095_50','infAp50','rbp095_500']
trec_exclude=['WT2012','WT2013','WT2014']
data_rbp095_500_topic_10_training=datasetload_training(column_names_rbp095_500_topic_10,path,trec_exclude)
data_rbp095_500_topic_15_training=datasetload_training(column_names_rbp095_500_topic_15,path,trec_exclude)
data_rbp095_500_topic_20_training=datasetload_training(column_names_rbp095_500_topic_20,path,trec_exclude)
data_rbp095_500_topic_25_training=datasetload_training(column_names_rbp095_500_topic_25,path,trec_exclude)
data_rbp095_500_topic_30_training=datasetload_training(column_names_rbp095_500_topic_30,path,trec_exclude)
data_rbp095_500_topic_35_training=datasetload_training(column_names_rbp095_500_topic_35,path,trec_exclude)
data_rbp095_500_topic_40_training=datasetload_training(column_names_rbp095_500_topic_40,path,trec_exclude)
data_rbp095_500_topic_45_training=datasetload_training(column_names_rbp095_500_topic_45,path,trec_exclude)
data_rbp095_500_topic_50_training=datasetload_training(column_names_rbp095_500_topic_50,path,trec_exclude)

#train Datasets Preparation
#precision
x_max=6
y=6
X_train_p1000_10_topicwise,Y_train_p1000_10_topicwise=train_dataset(data_p1000_topic_10_training,x_max,y)
X_train_p1000_15_topicwise,Y_train_p1000_15_topicwise=train_dataset(data_p1000_topic_15_training,x_max,y)
X_train_p1000_20_topicwise,Y_train_p1000_20_topicwise=train_dataset(data_p1000_topic_20_training,x_max,y)
X_train_p1000_25_topicwise,Y_train_p1000_25_topicwise=train_dataset(data_p1000_topic_25_training,x_max,y)
X_train_p1000_30_topicwise,Y_train_p1000_30_topicwise=train_dataset(data_p1000_topic_30_training,x_max,y)
X_train_p1000_35_topicwise,Y_train_p1000_35_topicwise=train_dataset(data_p1000_topic_35_training,x_max,y)
X_train_p1000_40_topicwise,Y_train_p1000_40_topicwise=train_dataset(data_p1000_topic_40_training,x_max,y)
X_train_p1000_45_topicwise,Y_train_p1000_45_topicwise=train_dataset(data_p1000_topic_45_training,x_max,y)
X_train_p1000_50_topicwise,Y_train_p1000_50_topicwise=train_dataset(data_p1000_topic_50_training,x_max,y)

X_train_p100_10_topicwise,Y_train_p100_10_topicwise=train_dataset(data_p100_topic_10_training,x_max,y)
X_train_p100_15_topicwise,Y_train_p100_15_topicwise=train_dataset(data_p100_topic_15_training,x_max,y)
X_train_p100_20_topicwise,Y_train_p100_20_topicwise=train_dataset(data_p100_topic_20_training,x_max,y)
X_train_p100_25_topicwise,Y_train_p100_25_topicwise=train_dataset(data_p100_topic_25_training,x_max,y)
X_train_p100_30_topicwise,Y_train_p100_30_topicwise=train_dataset(data_p100_topic_30_training,x_max,y)
X_train_p100_35_topicwise,Y_train_p100_35_topicwise=train_dataset(data_p100_topic_35_training,x_max,y)
X_train_p100_40_topicwise,Y_train_p100_40_topicwise=train_dataset(data_p100_topic_40_training,x_max,y)
X_train_p100_45_topicwise,Y_train_p100_45_topicwise=train_dataset(data_p100_topic_45_training,x_max,y)
X_train_p100_50_topicwise,Y_train_p100_50_topicwise=train_dataset(data_p100_topic_50_training,x_max,y)

X_train_p500_10_topicwise,Y_train_p500_10_topicwise=train_dataset(data_p500_topic_10_training,x_max,y)
X_train_p500_15_topicwise,Y_train_p500_15_topicwise=train_dataset(data_p500_topic_15_training,x_max,y)
X_train_p500_20_topicwise,Y_train_p500_20_topicwise=train_dataset(data_p500_topic_20_training,x_max,y)
X_train_p500_25_topicwise,Y_train_p500_25_topicwise=train_dataset(data_p500_topic_25_training,x_max,y)
X_train_p500_30_topicwise,Y_train_p500_30_topicwise=train_dataset(data_p500_topic_30_training,x_max,y)
X_train_p500_35_topicwise,Y_train_p500_35_topicwise=train_dataset(data_p500_topic_35_training,x_max,y)
X_train_p500_40_topicwise,Y_train_p500_40_topicwise=train_dataset(data_p500_topic_40_training,x_max,y)
X_train_p500_45_topicwise,Y_train_p500_45_topicwise=train_dataset(data_p500_topic_45_training,x_max,y)
X_train_p500_50_topicwise,Y_train_p500_50_topicwise=train_dataset(data_p500_topic_50_training,x_max,y)

#ndcg
X_train_ndcg1000_10_topicwise,Y_train_ndcg1000_10_topicwise=train_dataset(data_ndcg1000_topic_10_training,x_max,y)
X_train_ndcg1000_15_topicwise,Y_train_ndcg1000_15_topicwise=train_dataset(data_ndcg1000_topic_15_training,x_max,y)
X_train_ndcg1000_20_topicwise,Y_train_ndcg1000_20_topicwise=train_dataset(data_ndcg1000_topic_20_training,x_max,y)
X_train_ndcg1000_25_topicwise,Y_train_ndcg1000_25_topicwise=train_dataset(data_ndcg1000_topic_25_training,x_max,y)
X_train_ndcg1000_30_topicwise,Y_train_ndcg1000_30_topicwise=train_dataset(data_ndcg1000_topic_30_training,x_max,y)
X_train_ndcg1000_35_topicwise,Y_train_ndcg1000_35_topicwise=train_dataset(data_ndcg1000_topic_35_training,x_max,y)
X_train_ndcg1000_40_topicwise,Y_train_ndcg1000_40_topicwise=train_dataset(data_ndcg1000_topic_40_training,x_max,y)
X_train_ndcg1000_45_topicwise,Y_train_ndcg1000_45_topicwise=train_dataset(data_ndcg1000_topic_45_training,x_max,y)
X_train_ndcg1000_50_topicwise,Y_train_ndcg1000_50_topicwise=train_dataset(data_ndcg1000_topic_50_training,x_max,y)

X_train_ndcg100_10_topicwise,Y_train_ndcg100_10_topicwise=train_dataset(data_ndcg100_topic_10_training,x_max,y)
X_train_ndcg100_15_topicwise,Y_train_ndcg100_15_topicwise=train_dataset(data_ndcg100_topic_15_training,x_max,y)
X_train_ndcg100_20_topicwise,Y_train_ndcg100_20_topicwise=train_dataset(data_ndcg100_topic_20_training,x_max,y)
X_train_ndcg100_25_topicwise,Y_train_ndcg100_25_topicwise=train_dataset(data_ndcg100_topic_25_training,x_max,y)
X_train_ndcg100_30_topicwise,Y_train_ndcg100_30_topicwise=train_dataset(data_ndcg100_topic_30_training,x_max,y)
X_train_ndcg100_35_topicwise,Y_train_ndcg100_35_topicwise=train_dataset(data_ndcg100_topic_35_training,x_max,y)
X_train_ndcg100_40_topicwise,Y_train_ndcg100_40_topicwise=train_dataset(data_ndcg100_topic_40_training,x_max,y)
X_train_ndcg100_45_topicwise,Y_train_ndcg100_45_topicwise=train_dataset(data_ndcg100_topic_45_training,x_max,y)
X_train_ndcg100_50_topicwise,Y_train_ndcg100_50_topicwise=train_dataset(data_ndcg100_topic_50_training,x_max,y)

X_train_ndcg500_10_topicwise,Y_train_ndcg500_10_topicwise=train_dataset(data_ndcg500_topic_10_training,x_max,y)
X_train_ndcg500_15_topicwise,Y_train_ndcg500_15_topicwise=train_dataset(data_ndcg500_topic_15_training,x_max,y)
X_train_ndcg500_20_topicwise,Y_train_ndcg500_20_topicwise=train_dataset(data_ndcg500_topic_20_training,x_max,y)
X_train_ndcg500_25_topicwise,Y_train_ndcg500_25_topicwise=train_dataset(data_ndcg500_topic_25_training,x_max,y)
X_train_ndcg500_30_topicwise,Y_train_ndcg500_30_topicwise=train_dataset(data_ndcg500_topic_30_training,x_max,y)
X_train_ndcg500_35_topicwise,Y_train_ndcg500_35_topicwise=train_dataset(data_ndcg500_topic_35_training,x_max,y)
X_train_ndcg500_40_topicwise,Y_train_ndcg500_40_topicwise=train_dataset(data_ndcg500_topic_40_training,x_max,y)
X_train_ndcg500_45_topicwise,Y_train_ndcg500_45_topicwise=train_dataset(data_ndcg500_topic_45_training,x_max,y)
X_train_ndcg500_50_topicwise,Y_train_ndcg500_50_topicwise=train_dataset(data_ndcg500_topic_50_training,x_max,y)

#rbp
X_train_rbp095_1000_10_topicwise,Y_train_rbp095_1000_10_topicwise=train_dataset(data_rbp095_1000_topic_10_training,x_max,y)
X_train_rbp095_1000_15_topicwise,Y_train_rbp095_1000_15_topicwise=train_dataset(data_rbp095_1000_topic_15_training,x_max,y)
X_train_rbp095_1000_20_topicwise,Y_train_rbp095_1000_20_topicwise=train_dataset(data_rbp095_1000_topic_20_training,x_max,y)
X_train_rbp095_1000_25_topicwise,Y_train_rbp095_1000_25_topicwise=train_dataset(data_rbp095_1000_topic_25_training,x_max,y)
X_train_rbp095_1000_30_topicwise,Y_train_rbp095_1000_30_topicwise=train_dataset(data_rbp095_1000_topic_30_training,x_max,y)
X_train_rbp095_1000_35_topicwise,Y_train_rbp095_1000_35_topicwise=train_dataset(data_rbp095_1000_topic_35_training,x_max,y)
X_train_rbp095_1000_40_topicwise,Y_train_rbp095_1000_40_topicwise=train_dataset(data_rbp095_1000_topic_40_training,x_max,y)
X_train_rbp095_1000_45_topicwise,Y_train_rbp095_1000_45_topicwise=train_dataset(data_rbp095_1000_topic_45_training,x_max,y)
X_train_rbp095_1000_50_topicwise,Y_train_rbp095_1000_50_topicwise=train_dataset(data_rbp095_1000_topic_50_training,x_max,y)

X_train_rbp095_100_10_topicwise,Y_train_rbp095_100_10_topicwise=train_dataset(data_rbp095_100_topic_10_training,x_max,y)
X_train_rbp095_100_15_topicwise,Y_train_rbp095_100_15_topicwise=train_dataset(data_rbp095_100_topic_15_training,x_max,y)
X_train_rbp095_100_20_topicwise,Y_train_rbp095_100_20_topicwise=train_dataset(data_rbp095_100_topic_20_training,x_max,y)
X_train_rbp095_100_25_topicwise,Y_train_rbp095_100_25_topicwise=train_dataset(data_rbp095_100_topic_25_training,x_max,y)
X_train_rbp095_100_30_topicwise,Y_train_rbp095_100_30_topicwise=train_dataset(data_rbp095_100_topic_30_training,x_max,y)
X_train_rbp095_100_35_topicwise,Y_train_rbp095_100_35_topicwise=train_dataset(data_rbp095_100_topic_35_training,x_max,y)
X_train_rbp095_100_40_topicwise,Y_train_rbp095_100_40_topicwise=train_dataset(data_rbp095_100_topic_40_training,x_max,y)
X_train_rbp095_100_45_topicwise,Y_train_rbp095_100_45_topicwise=train_dataset(data_rbp095_100_topic_45_training,x_max,y)
X_train_rbp095_100_50_topicwise,Y_train_rbp095_100_50_topicwise=train_dataset(data_rbp095_100_topic_50_training,x_max,y)

X_train_rbp095_500_10_topicwise,Y_train_rbp095_500_10_topicwise=train_dataset(data_rbp095_500_topic_10_training,x_max,y)
X_train_rbp095_500_15_topicwise,Y_train_rbp095_500_15_topicwise=train_dataset(data_rbp095_500_topic_15_training,x_max,y)
X_train_rbp095_500_20_topicwise,Y_train_rbp095_500_20_topicwise=train_dataset(data_rbp095_500_topic_20_training,x_max,y)
X_train_rbp095_500_25_topicwise,Y_train_rbp095_500_25_topicwise=train_dataset(data_rbp095_500_topic_25_training,x_max,y)
X_train_rbp095_500_30_topicwise,Y_train_rbp095_500_30_topicwise=train_dataset(data_rbp095_500_topic_30_training,x_max,y)
X_train_rbp095_500_35_topicwise,Y_train_rbp095_500_35_topicwise=train_dataset(data_rbp095_500_topic_35_training,x_max,y)
X_train_rbp095_500_40_topicwise,Y_train_rbp095_500_40_topicwise=train_dataset(data_rbp095_500_topic_40_training,x_max,y)
X_train_rbp095_500_45_topicwise,Y_train_rbp095_500_45_topicwise=train_dataset(data_rbp095_500_topic_45_training,x_max,y)
X_train_rbp095_500_50_topicwise,Y_train_rbp095_500_50_topicwise=train_dataset(data_rbp095_500_topic_50_training,x_max,y)


#This is a section for load the test datasets for high cost precision metric

trec_include=['WT2012']
# test data for precision at 1000 at depths varying from 10 to 50
test_data_2012_p1000_topic_10=datasetload_testing(path,column_names_p1000_topic_10,trec_include)
test_data_2012_p1000_topic_15=datasetload_testing(path,column_names_p1000_topic_15,trec_include)
test_data_2012_p1000_topic_20=datasetload_testing(path,column_names_p1000_topic_20,trec_include)
test_data_2012_p1000_topic_25=datasetload_testing(path,column_names_p1000_topic_25,trec_include)
test_data_2012_p1000_topic_30=datasetload_testing(path,column_names_p1000_topic_30,trec_include)
test_data_2012_p1000_topic_35=datasetload_testing(path,column_names_p1000_topic_35,trec_include)
test_data_2012_p1000_topic_40=datasetload_testing(path,column_names_p1000_topic_40,trec_include)
test_data_2012_p1000_topic_45=datasetload_testing(path,column_names_p1000_topic_45,trec_include)
test_data_2012_p1000_topic_50=datasetload_testing(path,column_names_p1000_topic_50,trec_include)
# test data for precision at 100 at depths varying from 10 to 50
test_data_2012_p100_topic_10=datasetload_testing(path,column_names_p100_topic_10,trec_include)
test_data_2012_p100_topic_15=datasetload_testing(path,column_names_p100_topic_15,trec_include)
test_data_2012_p100_topic_20=datasetload_testing(path,column_names_p100_topic_20,trec_include)
test_data_2012_p100_topic_25=datasetload_testing(path,column_names_p100_topic_25,trec_include)
test_data_2012_p100_topic_30=datasetload_testing(path,column_names_p100_topic_30,trec_include)
test_data_2012_p100_topic_35=datasetload_testing(path,column_names_p100_topic_35,trec_include)
test_data_2012_p100_topic_40=datasetload_testing(path,column_names_p100_topic_40,trec_include)
test_data_2012_p100_topic_45=datasetload_testing(path,column_names_p100_topic_45,trec_include)
test_data_2012_p100_topic_50=datasetload_testing(path,column_names_p100_topic_50,trec_include)
# test data for precision at 500 at depths varying from 10 to 50
test_data_2012_p500_topic_10=datasetload_testing(path,column_names_p500_topic_10,trec_include)
test_data_2012_p500_topic_15=datasetload_testing(path,column_names_p500_topic_15,trec_include)
test_data_2012_p500_topic_20=datasetload_testing(path,column_names_p500_topic_20,trec_include)
test_data_2012_p500_topic_25=datasetload_testing(path,column_names_p500_topic_25,trec_include)
test_data_2012_p500_topic_30=datasetload_testing(path,column_names_p500_topic_30,trec_include)
test_data_2012_p500_topic_35=datasetload_testing(path,column_names_p500_topic_35,trec_include)
test_data_2012_p500_topic_40=datasetload_testing(path,column_names_p500_topic_40,trec_include)
test_data_2012_p500_topic_45=datasetload_testing(path,column_names_p500_topic_45,trec_include)
test_data_2012_p500_topic_50=datasetload_testing(path,column_names_p500_topic_50,trec_include)
#test set for trec 2013 follows below
trec_include=['WT2013']
# test data for precision at 1000 at depths varying from 10 to 50
test_data_2013_p1000_topic_10=datasetload_testing(path,column_names_p1000_topic_10,trec_include)
test_data_2013_p1000_topic_15=datasetload_testing(path,column_names_p1000_topic_15,trec_include)
test_data_2013_p1000_topic_20=datasetload_testing(path,column_names_p1000_topic_20,trec_include)
test_data_2013_p1000_topic_25=datasetload_testing(path,column_names_p1000_topic_25,trec_include)
test_data_2013_p1000_topic_30=datasetload_testing(path,column_names_p1000_topic_30,trec_include)
test_data_2013_p1000_topic_35=datasetload_testing(path,column_names_p1000_topic_35,trec_include)
test_data_2013_p1000_topic_40=datasetload_testing(path,column_names_p1000_topic_40,trec_include)
test_data_2013_p1000_topic_45=datasetload_testing(path,column_names_p1000_topic_45,trec_include)
test_data_2013_p1000_topic_50=datasetload_testing(path,column_names_p1000_topic_50,trec_include)
# test data for precision at 100 at depths varying from 10 to 50
test_data_2013_p100_topic_10=datasetload_testing(path,column_names_p100_topic_10,trec_include)
test_data_2013_p100_topic_15=datasetload_testing(path,column_names_p100_topic_15,trec_include)
test_data_2013_p100_topic_20=datasetload_testing(path,column_names_p100_topic_20,trec_include)
test_data_2013_p100_topic_25=datasetload_testing(path,column_names_p100_topic_25,trec_include)
test_data_2013_p100_topic_30=datasetload_testing(path,column_names_p100_topic_30,trec_include)
test_data_2013_p100_topic_35=datasetload_testing(path,column_names_p100_topic_35,trec_include)
test_data_2013_p100_topic_40=datasetload_testing(path,column_names_p100_topic_40,trec_include)
test_data_2013_p100_topic_45=datasetload_testing(path,column_names_p100_topic_45,trec_include)
test_data_2013_p100_topic_50=datasetload_testing(path,column_names_p100_topic_50,trec_include)
# test data for precision at 500 at depths varying from 10 to 50
test_data_2013_p500_topic_10=datasetload_testing(path,column_names_p500_topic_10,trec_include)
test_data_2013_p500_topic_15=datasetload_testing(path,column_names_p500_topic_15,trec_include)
test_data_2013_p500_topic_20=datasetload_testing(path,column_names_p500_topic_20,trec_include)
test_data_2013_p500_topic_25=datasetload_testing(path,column_names_p500_topic_25,trec_include)
test_data_2013_p500_topic_30=datasetload_testing(path,column_names_p500_topic_30,trec_include)
test_data_2013_p500_topic_35=datasetload_testing(path,column_names_p500_topic_35,trec_include)
test_data_2013_p500_topic_40=datasetload_testing(path,column_names_p500_topic_40,trec_include)
test_data_2013_p500_topic_45=datasetload_testing(path,column_names_p500_topic_45,trec_include)
test_data_2013_p500_topic_50=datasetload_testing(path,column_names_p500_topic_50,trec_include)
#test set for trec 2014 follows below
trec_include=['WT2014']
# test data for precision at 1000 at depths varying from 10 to 50
test_data_2014_p1000_topic_10=datasetload_testing(path,column_names_p1000_topic_10,trec_include)
test_data_2014_p1000_topic_15=datasetload_testing(path,column_names_p1000_topic_15,trec_include)
test_data_2014_p1000_topic_20=datasetload_testing(path,column_names_p1000_topic_20,trec_include)
test_data_2014_p1000_topic_25=datasetload_testing(path,column_names_p1000_topic_25,trec_include)
test_data_2014_p1000_topic_30=datasetload_testing(path,column_names_p1000_topic_30,trec_include)
test_data_2014_p1000_topic_35=datasetload_testing(path,column_names_p1000_topic_35,trec_include)
test_data_2014_p1000_topic_40=datasetload_testing(path,column_names_p1000_topic_40,trec_include)
test_data_2014_p1000_topic_45=datasetload_testing(path,column_names_p1000_topic_45,trec_include)
test_data_2014_p1000_topic_50=datasetload_testing(path,column_names_p1000_topic_50,trec_include)
# test data for precision at 100 at depths varying from 10 to 50
test_data_2014_p100_topic_10=datasetload_testing(path,column_names_p100_topic_10,trec_include)
test_data_2014_p100_topic_15=datasetload_testing(path,column_names_p100_topic_15,trec_include)
test_data_2014_p100_topic_20=datasetload_testing(path,column_names_p100_topic_20,trec_include)
test_data_2014_p100_topic_25=datasetload_testing(path,column_names_p100_topic_25,trec_include)
test_data_2014_p100_topic_30=datasetload_testing(path,column_names_p100_topic_30,trec_include)
test_data_2014_p100_topic_35=datasetload_testing(path,column_names_p100_topic_35,trec_include)
test_data_2014_p100_topic_40=datasetload_testing(path,column_names_p100_topic_40,trec_include)
test_data_2014_p100_topic_45=datasetload_testing(path,column_names_p100_topic_45,trec_include)
test_data_2014_p100_topic_50=datasetload_testing(path,column_names_p100_topic_50,trec_include)
# test data for precision at 500 at depths varying from 10 to 50
test_data_2014_p500_topic_10=datasetload_testing(path,column_names_p500_topic_10,trec_include)
test_data_2014_p500_topic_15=datasetload_testing(path,column_names_p500_topic_15,trec_include)
test_data_2014_p500_topic_20=datasetload_testing(path,column_names_p500_topic_20,trec_include)
test_data_2014_p500_topic_25=datasetload_testing(path,column_names_p500_topic_25,trec_include)
test_data_2014_p500_topic_30=datasetload_testing(path,column_names_p500_topic_30,trec_include)
test_data_2014_p500_topic_35=datasetload_testing(path,column_names_p500_topic_35,trec_include)
test_data_2014_p500_topic_40=datasetload_testing(path,column_names_p500_topic_40,trec_include)
test_data_2014_p500_topic_45=datasetload_testing(path,column_names_p500_topic_45,trec_include)
test_data_2014_p500_topic_50=datasetload_testing(path,column_names_p500_topic_50,trec_include)

