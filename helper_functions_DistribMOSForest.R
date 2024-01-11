library(model4you)
library(crch)
library(disttree)

# Function to train DistribForest 
DistribMOSForest_train_forest<-function(df_train, list_hp){
    ###-----------------------------------------------------------------------------
    ###Input
    #df_train.........Train data (data.frame) including predictors and obs. information.  (n_train x (n_preds + 1) data.frame)
    #list_hp..........List of hyperparameters for the DistribMOSForest model.
    #.................The list should at least contain the names of predictors to use (vector names_pred, including "ens_mean"), 
    #.................the distribution family to fit (family_="gaussian" or "logistic"),
    #.................the number of trees in the forest (nb_trees=100 is preferable)
    #.................the number of variables to possibly split at in each node (mtry),
    #.................the minimal node size to consider further split (minsplit=50 is ok).
    #.................the minimal terminal node size allowed (min_node_size = 20 is ok)
    #See package disttree for further details.
    ###-----------------------------------------------------------------------------
    ###Output
    #res..............List containing:
    #df_train.........Train data used as inputs.
    #forest_train.....The fitted forest (distforest object)
    #list_param.......List of hyperparameters for the distforest model.
    ###------------------------------------                                                           
  
    print('Training DistribMOSForest...')
    #Select predictors
    df_train=df_train[,c("obs",list_hp$names_pred)]
    ## Training MOS forest using crch with "truncated = TRUE" (i.e., trch)
    tmp_family=list_hp$family 
    if(tmp_family=="gaussian"){
        mos <- crch(obs ~ ens_mean, data = df_train, truncated=TRUE, left = 0, dist="gaussian")
    }
    if(tmp_family=="logistic"){
        mos <- crch(obs ~ ens_mean, data = df_train, truncated=TRUE, left = 0, dist="logistic")
    }

    #Fit Distributional MOS Forest
    mos_forest <- pmforest(mos, 
              data = df_train, 
              ntree =  list_hp$nb_trees,
              type.tree = "ctree",
              control = list_hp$ctree_ctrl, trace=TRUE)

  return(list(df_train=df_train, 
              forest_train=mos_forest, 
              list_param=list_hp))
}



# Function to predict with forest_train from DistribMOSForest_train_forest() 
DistribMOSForest_predict_forest<-function(forest_train_object, df_test, list_hp, probs_to_draw=1:51/52, add_info=TRUE){
    ###-----------------------------------------------------------------------------
    ###Input
    #forest_train_object.........Trained forest object obtained from DistribMOSForest_train_forest().
    #df_test.........Test data used as inputs.
    #list_hp..........List of hyperparameters for the DistribMOSForest model.
    #.................The list should at least contain the names of predictors to use (vector names_pred), 
    #.................the distribution family to consider (family_),
    #See package disttree for further details.
    #probs_to_draw............Probabilities for which quantiles have to be estimated.
    #add_info.................Boolean. Should information from the test dataset be
    #.........................added to the final output? =FALSE if 
    #.........................different data than those from KNMI are used (general case).
    ###-----------------------------------------------------------------------------
    ###Output
    #res..............List containing:
    #df_PP.................Data.frame containing DistribForest post-processed data in ens1,...,ens51.
    #df_test...............Test data used as input.
    #list_hp...............List of hyperparameters for the DistribForest
    ###-----------------------------------------------------------------------------
    print('Predict...')
    nb_members=length(probs_to_draw)
    # Compute outputs for test set
    tmp_res=matrix(NaN, ncol=nb_members, nrow=length(df_test[,1]))

    ### Prediction on test set
    #Select predictors
    predictor_df_test=(df_test[,list_hp$names_pred])
    pred.hat <- pmodel(x = forest_train_object, newdata = predictor_df_test, fun = identity)    

    # Make predictions of distributional parameters
    p <- do.call(rbind,
                lapply(names(pred.hat),
                        function(x) {predict(pred.hat[[x]],
                                            newdata = predictor_df_test[x,],
                                            type = "parameter")}))
    if(list_hp$family %in% c("logistic", "gaussian")){
        fitted_mu <- p$location
        fitted_sigma <- p$scale
        for(i in 1:nrow(predictor_df_test)){
            if(list_hp$family=="logistic"){tmp_res[i,]<-qtlogis(probs_to_draw, location=fitted_mu[i], scale=fitted_sigma[i], left=0)}
            if(list_hp$family=="gaussian"){tmp_res[i,]<-qtnorm(probs_to_draw, mean=fitted_mu[i], sd=fitted_sigma[i], left=0)}
        }
    }
    tmp_res=as.data.frame(tmp_res)
    colnames(tmp_res)=paste0("ens", 1:length(probs_to_draw))
    if(add_info==TRUE){
        final_output<-add_info_to_dataframe(tmp_res, df_test)
    }else{
        final_output <- tmp_res
    }
    print("Done")
    return(list(df_PP=final_output,
              df_test=df_test, 
              list_param=list_hp))
}


