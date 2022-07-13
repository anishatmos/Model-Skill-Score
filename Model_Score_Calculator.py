
"""
This Program is calculate the three hourly model scores for Bihar lightning case using precipication data

API - WORKFLOW 

1.  Insert Input Data --- 

    Observations: Observation.csv (ERA5 Reanalysis data of Total Precipication [tp] having resolution 25 KM).
    Model's Output Data: Model_Output_D01.csv, Model_Output_D02.csv, Model_Output_D03.csv (These are the WRF model simulated to
                         output for different domain such as Domain 01 [D01] [D02] [D03], Moreover every domain have different resolution 9 KM, 3 KM, 1 KM respectively).

2. Find Nearest Neighbour from Model's Output Data w.r.t. Observations because every data have a differnt resolution (i.e. every input data files consist different number of values).  

3. Step 2 will give us two dataset - one observation & second - Preprocessed model output data.

4. Segregate the data for each time steps for observations & Model Output because input data is 3 hourly (i.e. data avaliable for 00,03,06,09,12,15,18,21 hours in a day).

5. Use confusion matrix to get the Contigency Table (matrix score) for evey time steps (Confusion matrix works on basis of threshold value of tp).

6. Using the value of contigency table, model scores (Accuracy, Threat Score, Bias, ETS) have been calculated for each time step.

7. Plot graphs for each time step.

"""

"""
Imports Important Libraries 

pandas for data analysis
sklearn.metrics for confusion matrix (Contigency Table) used to caluate the score
plotly.graph_objects for plot the calculated model score

"""

import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go




def generate_data(observation,model_output):
    """
    This function will generate two dataset that has same length using nearest neighbour.

    Input:
    Observation: Actual Observation Data [ ERA5 Reanalysis data]
    Model_Output : Predicted data from WRF model for different Domain [Domain 01, 02, 03]

    Output:
    It will return dataset - actual, pred

    actual : preprocessed observation data
    pred : preprocessed model output data
    """

    actual_lat=observation['lat'].unique()    
    pred_lat=model_output['lat'].unique()     
    actual_lat_list=[]
    predict_lat_list=[]
    for i in range(len(actual_lat)):
        actual_lat_list.append(actual_lat[i])
        distance=list(abs(pred_lat-actual_lat[i]))
        index=distance.index(min(distance))
        predict_lat_list.append(pred_lat[index])
    
    actual_lon=observation['lon'].unique()
    pred_lon=model_output['lon'].unique()

    actual_lon_list=[]
    predict_lon_list=[]

    for i in range(len(actual_lon)):
        actual_lon_list.append(actual_lon[i])
        distance=list(abs(pred_lon-actual_lon[i]))
        index=distance.index(min(distance))
        predict_lon_list.append(pred_lon[index])

    pred=model_output[model_output.lat.isin(predict_lat_list)]    
    pred=pred[pred.lon.isin(predict_lon_list)]


    actual=observation.sort_values(['lat','lon'])
    pred=pred.sort_values(['lat','lon'])

    return actual, pred





def generate_score(actual_, pred_,threshold):
    """
    This function will take preprocessed actual and model output data and will return matrix value
    for each time step for some threshold value along with list of time steps

    Input:
    actual_ : preprocessed observation data
    pred_ : preprocessed model output data
    threshold : threshold value to use for matrix calculation [used value 2.5 mm and 5 mm]

    Output:
    matrix score : matrix Score that contains the values of contigency table.
    time_list : List of unique time steps for which metric score is calculated

    """

    time_list=list(actual_.time.unique())
    actual_group=actual_.groupby('time')
    pred_group=pred_.groupby('time')

    score_list=[]
    for i in time_list:
        actual_pred=actual_group.get_group(i)['tp'].apply(lambda x: 1 if x<=threshold else 0)
        pred_pred=pred_group.get_group(i)['tp'].apply(lambda x: 1 if x<=threshold else 0)
        cm=confusion_matrix(actual_pred,pred_pred)
        score_list.append(calculate_metrics(cm))       #calculate_metrics is a helper fn to calculate matrix based on cm


    score=pd.DataFrame(score_list)
    score.columns=['Accuracy', 'Threat Score', 'Bias','ETS']

    return score, time_list




def calculate_metrics(conf_metr):
    """
    This fucntion takes confusion matrix as input and generate different model score such as Accuracy, Threat Score, Bias, ETS
    
    Input :
    conf_metr : Confusion matrix (Matrix score)

    Output : Model score
    Accuracy : What fraction of the forecasts were correct Range:0 to 1. Perfect score:1
   
    Threat_score : (Critical Sucess Index) How well did the forecast "yes" events correspond to the observed "yes" events. Range: 0-1, 0 indicates no skill, 1 represents perect score
    
    Bias : (or frequency Bias) How similar were the frequencies of Yes forecasts and Yes observations? Range:0 to infinity. Perfect score:1 When Bias is greater than 1, the event is overforecast; less than 1,   	  	   underforecast.

    ETS : How well did the forecast "yes" events correspond to the observed "yes" events (accounting for hits that would be expected by chance
 	  range: -1/3-1, 0 indicates no skills, 1 is perfect score. It is the number of hits for random forecasts
    
    """
    NN=conf_metr[0,0]
    YY=conf_metr[1,1]
    YN=conf_metr[1,0]
    NY=conf_metr[0,1]

    #Accuracy
    accuracy=(YY+NN)/(YY+NN+NY+YN)

    #Threat Score
    threat_score=YY/(YY+NY+YN)

    #Bias
    bias=(YY+NY)/(YY+YN)

    #Equitable Threat Score

    YY_random=((YY+NY)*(YY+YN))/(YY+NY+YN+NN)
    ets=(YY-YY_random)/(YY+YN+NY-YY_random)

    return accuracy,threat_score,bias,ets




def generate_plot(actual_path, pred_path, threshold_list,metric):
    """
    This Function will generate plot for specified list of threshold and metric for different model's output data

    Input:
    actual_path : path of the observation data file [Observation.csv]
    pred_path : List of paths for different different model's output data file [Model_Output_D01.csv, Model_Output_D02.csv, Model_Output_D03.csv]
    threshold_list : list of threshold value [used value 2.5 mm and 5 mm]
    metric : Name of model score for which you want to generate plot [Accuracy, Threat Score, Bias, ETS]
    
    return :
    Return a plotly figure of generated plot
    """
    
    fig=go.Figure()
    fig.update_layout({'plot_bgcolor':'#FFFFFF', 'paper_bgcolor': '#f6fcfc'},modebar=dict(bgcolor='rgba(34,34,34,0.6)'))

    for threshold in threshold_list:
        for pred_path_ in pred_path:
            actual=pd.read_csv(actual_path)
            pred=pd.read_csv(pred_path_)
            actual_, pred_=generate_data(actual,pred)
            score,time_list=generate_score(actual_,pred_,threshold)
            fig=add_plot(fig,time_list,score,metric,pred_path_,threshold)   #add_plot is a helper function to add plot in Main plot figure
    return fig


def add_plot(fig,time_list, score, variable,pred_path,value):

    """
    A helper function to add plot in Main plot
    """

    name=str(pred_path.split("_")[2][:-4])+"_"+str(value)+"mm"
    fig.add_trace(go.Scatter(x=time_list,y=score[variable],name=name))
    fig.update_layout(title={'text':" Three Hourly {""}".format(variable),'y':0.95,'x':0.5,'xanchor':'center','yanchor':'top'},font=dict(family='Courier New, monospace',size=22, color='RebeccaPurple'))
    fig.update_xaxes(title="Time",dtick=time_list)
    fig.update_yaxes(title=str(variable))
    

    return fig

fig1=generate_plot('Observation.csv',['Model_Output_D01.csv','Model_Output_D02.csv','Model_Output_D03.csv'],[2.5,5],'Accuracy')
fig2=generate_plot('Observation.csv',['Model_Output_D01.csv','Model_Output_D02.csv','Model_Output_D03.csv'],[2.5,5],'Threat Score')
fig3=generate_plot('Observation.csv',['Model_Output_D01.csv','Model_Output_D02.csv','Model_Output_D03.csv'],[2.5,5],'Bias')
fig4=generate_plot('Observation.csv',['Model_Output_D01.csv','Model_Output_D02.csv','Model_Output_D03.csv'],[2.5,5],'ETS')


# These line of codes store plotly graph as html file
fig1.write_html('Accuracy.html')
fig2.write_html('Threat_score.html')
fig3.write_html('Bias.html')
fig4.write_html('ETS.html')

print("\nRun successfully.\n\nPlease Find The Following Output Files in Your Directory.\n1) Accuracy.html\n2) Threat_Score.html\n3) Bias.html\n4) ETS.html")

