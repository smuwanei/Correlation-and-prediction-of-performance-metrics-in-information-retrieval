#This section contains functions
def datasetload_training(column_names,path,trec_exclude):
    #load dataset
    topicWiseDataset=read_csv(path)
    training_set=topicWiseDataset[~topicWiseDataset.trec.isin(trec_exclude)]
    data=training_set[column_names]
    return data

def datasetload_testing(path,column_names,trec_include):
    topicWiseDataset=read_csv(path)
    test_set=topicWiseDataset[topicWiseDataset.trec.isin(trec_include)]
    data=test_set[column_names]
    return data
def descriptive_statistics(data,type_wise):
    print(type_wise)
    print("1.dimensions of dataset")
    print(data.shape)
    print("2.data types for dataset attributes")
    print(data.dtypes)
    print("3.peek the first 20 rows")
    print(data.head(20))
    set_option('precision',1)
    print("4.descriptions of dataset attributes")
    print(data.describe())
    
def data_visualization(data,column_names,type_wise):
    print(type_wise)
    print("Unimodal Visualizations")
    print("1.Histograms")
    data.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
    pyplot.show()
    print("2.Density Plots")
    data.plot(kind='density',subplots=True,layout=(4,4),sharex=False,legend=False,fontsize=1)
    pyplot.show()
    print("3.Box Plots")
    data.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=8)
    pyplot.show()
    print("Multimodal Visualizations")
    print("4.Scatter matrix")
    scatter_matrix(data)
    pyplot.show()
    print("5.Correlation Matrix")
    fig=pyplot.figure()
    ax=fig.add_subplot(111)
    cax=ax.matshow(data.corr(),vmin=-1,vmax=1,interpolation='none')
    fig.colorbar(cax)
    ticks=numpy.arange(0,7,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(column_names)
    ax.set_yticklabels(column_names)
    pyplot.show()
    
def train_validation_split(data,x_max,y,seed,validation_size):
    array=data.values
    X=array[:,0:x_max]
    Y=array[:,y]
    validation_size=validation_size
    seed=seed
    X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)
    return X_train,X_validation,Y_train,Y_validation

def test_dataset(data,x_max,y):
    array=data.values
    x_test=array[:,0:x_max]
    y_test=array[:,y]
    return x_test,y_test

def train_dataset(data,x_max,y):
    array=data.values
    x_train=array[:,0:x_max]
    y_train=array[:,y]
    return x_train,y_train
    

def prediction(model,x_test_data):
    scaler=StandardScaler().fit(x_test_data)
    rescaledX=scaler.transform(x_test_data)
    y_predicted=model.predict(rescaledX)
    return y_predicted
def tau(y_predicted,y_test):
    tau_value,p_value=kendalltau(y_test,y_predicted)
    return tau_value
def spearman(y_predicted,y_test):
    spearman_value,p_value=spearmanr(y_test,y_predicted)
    return spearman_value
    
def graph_tau(depth,precision_list_topic,precision_list_system,ndcg_list_topic,ndcg_list_system,precision_label_topic,precision_label_system,ndcg_label_topic,ndcg_label_system,xlabel,ylabel,title,save_path):
    
    #now follows plotting of graphs
    pyplot.plot(depth,precision_list_topic,label=precision_label_topic,linestyle='solid',marker='+')
    pyplot.plot(depth,precision_list_system,label=precision_label_topic,linestyle='solid',marker='d')
    pyplot.plot(depth,ndcg_list_topic,label=ndcg_label_topic,linestyle='solid',marker='x')
    pyplot.plot(depth,ndcg_list_system,label=ndcg_label_system,linestyle='solid',marker='^')
    pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    #save graph to specific folder
    pyplot.savefig(save_path)

    
def graph_tau(depth,precision_list_topic,precision_list_system,ndcg_list_topic,ndcg_list_system,precision_label_topic,precision_label_system,ndcg_label_topic,ndcg_label_system,xlabel,ylabel,title,save_path):
    
    #now follows plotting of graphs
    pyplot.plot(depth,precision_list_topic,label=precision_label_topic,linestyle='solid',marker='+')
    pyplot.plot(depth,precision_list_system,label=precision_label_topic,linestyle='solid',marker='d')
    pyplot.plot(depth,ndcg_list_topic,label=ndcg_label_topic,linestyle='solid',marker='x')
    pyplot.plot(depth,ndcg_list_system,label=ndcg_label_system,linestyle='solid',marker='^')
    pyplot.legend()
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    #save graph to specific folder
    pyplot.savefig(save_path)
    
    
def dataframe_metric(list_metrics,list_col_names):
    df=pd.DataFrame(list_metrics,list_col_names)
    return df

# functions relating to deep neural nets begin here
#fit model with given number of nodes,returns test set accuracy and evaluate best dropout rate 

def dnn_baseline_model(trainX,trainy):
    #configure the model based on the data
    #n_input,n_classes=trainX.shape[1],testy.shape[1]
    #define model
    n_input=6
    n_nodes=5
    dropout_rate=0.1
    #n_nodes=int(n_nodes/dropout_rate)
    model=Sequential()
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_nodes,input_dim=n_input,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='linear'))
    #compile model
    opt=SGD(lr=0.01,momentum=0.9,decay=decay)
    model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mse'])
    #fit model on train set
    model.fit(trainX,trainy,epochs=200)
    return model

#below is creation of folder for stacked ensemble models
#Replace this path to where your deep learning models will be stored prior to ensembling
#path_model='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\models'
#makedirs(path_model)
#this function is for creating submodels
def create_sub_models(deep_model,n_members,metric_depth,x_train,y_train):
    for i in range(n_members):
        #fit model
        model=KerasRegressor(build_fn=deep_model,batch_size=32,epochs=300)
        model.fit(x_train,y_train)
        #save model
        path_model='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\models'
        filename=path_model+'\\model_'+str(i+1)+metric_depth+'.h5'
        model.model.save(filename)
        print('>Saved %s' % filename)
    
#This function below is for loading saved models
def load_all_models(deep_model,n_models,metric_depth):
    all_models=list()
    for i in range(n_models):
        model=KerasRegressor(build_fn=deep_model,batch_size=32,epochs=300)
        #define file name for this essemble
        path_model='C:\\Users\\Intern Student\\Desktop\\PhdWork_IR\\secondPaper\\models'
        filename=path_model+'\\model_'+str(i+1)+metric_depth+'.h5'
        #load model from file
        model.model=load_model(filename)
        #add to list of members
        all_models.append(model)
        print('>loaded %s' %filename)
    return all_models
# create a stacked model input dataset as outpute from the ensemble
def stacked_dataset(members,inputX):
    stackX=None
    for model in members:
        #make prediction
        yhat=model.predict(inputX,verbose=0)
        #stack predictions into [rows,members,probabilities]
        if stackX is None:
            stackX=yhat
        else:
            stackX=dstack((stackX,yhat))
    #flatten predictions to [rows,members x probabilities]
    stackX=stackX.reshape((stackX.shape[0],stackX.shape[1]*stackX.shape[2]))
    return stackX
#fit a model based on the outputs from the ensemble members
def fit_stacked_model(members,inputX,inputy):
    #create a dataset using ensemble
    stackedX=stacked_dataset(members,inputX)
    #fit stand alone model
    model=LinearRegression()
    model.fit(stackedX,inputy)
    return model
#make prediction with stacked model
def stacked_prediction(members,model,inputX):
    #create dataset using ensemble
    stackedX=stacked_dataset(members,inputX)
    #make a prediction
    yhat=model.predict(stackedX)
    return yhat

#Define stacked model from multiple member input models
def define_stacked_model(members,path):
    #update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='linear')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file=path)
    # compile
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    #inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(X, inputy, epochs=300, verbose=0)
    
# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)
            
        
        
