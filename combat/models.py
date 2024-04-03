# -*- coding: utf-8 -*-
"""
The module Models initiates the LogitModel class
"""





import pandas as pd
import numpy as np
from typing import Union, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score
                            , precision_score
                            , recall_score
                            , f1_score
                            , auc
                            , roc_curve
                            , accuracy_score
                            , brier_score_loss
                            , confusion_matrix
                            )

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

import matplotlib.pyplot as plt
from typing import Optional, Union

import warnings
warnings.filterwarnings("ignore")


class LogitModel:
    sk_model = None
    sm_model = None
    
    def __init__(self
                , x_train: pd.DataFrame
                , y_train: pd.Series
                , x_test: pd.DataFrame
                , y_test: pd.Series
                , intercept: bool = True
                , penalty: Optional[str] = None
                , alpha: float = 0.1
                ):
        
        # =============================================================================
        # Input Data Validation        
        # =============================================================================
        
        if not isinstance(x_train, pd.DataFrame):
            raise TypeError("""The 'x_train' parameter must be pd.DataFrame""")

        if not isinstance(y_train, pd.Series):
            raise TypeError("""The 'y_train' parameter must be pd.Series""")
        
        if not isinstance(x_test, pd.DataFrame):
            raise TypeError("""The 'x_test' parameter must be pd.DataFrame""")
        
        if not isinstance(y_test, pd.Series):
            raise TypeError("""The 'y_test' parameter must be pd.Series""")
            
        if not x_train.columns.equals(x_test.columns):
            raise ValueError("""Columns of 'x_train' and 'x_test' are not identical""")
            
        if len(x_train) != len(y_train):
            raise ValueError("""The length of 'x_train' and 'y_train' must be identical""")
            
        if len(x_test) != len(y_test):
            raise ValueError("""The length of 'x_test' and 'y_test' must be identical""")
            
        if not isinstance(intercept, bool):
            raise ValueError("""The 'intercept' parameter must be logical""")
            
        if penalty not in [None, 'l1']:
            raise ValueError("""The 'penalty' parameter must be iether None or 'l1'; got {}""".format(penalty))
            
        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError("""The 'alpha' parameter must be float from 0 to 1; got {}""".format(alpha))
                        
        # =============================================================================
        # initializing    
        # =============================================================================
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.penalty = penalty
        self.intercept = intercept
        self.alpha = alpha       
    
    def Model_SK(self):
        if self.sk_model is None:
            model = LogisticRegression(
                          penalty = self.penalty
                        , fit_intercept = self.intercept
                        # , l1_ratio = self.l1_ratio
                        , solver = 'saga'
                        , C = self.alpha
                    ).fit(self.x_train, self.y_train)
            self.sk_model = model
        
            return model
        else:
            return None
        
    def Model_SM(self):
        if self.sm_model is None:
            if self.intercept:
                x_train_sm = add_constant(self.x_train)
            else: 
                x_train_sm = self.x_train
                
            if self.penalty == 'l1':
                model  = sm.Logit(self.y_train, x_train_sm).fit_regularized(method = self.penalty
                                                                            , alpha=self.alpha
                                                                            , disp=False
                                                                            )
            else:
                model = sm.Logit(self.y_train, x_train_sm).fit(disp = False
                                                               , method = 'newton')
            
            self.sm_model = model
            
            return model
        else:
            return None
        
    # =============================================================================
    # Train Sample Metrics        
    # =============================================================================
    def Gini_Train(self) -> float:
        """
        Calculates Gini on the Train set
        """
        
        gini = 2 * self.AUC_Train() - 1
        
        gini = float("{:.3f}".format(gini))
        return gini
        
    def Accuracy_Train(self
                       , cutoff: float
                       ) -> float:
        """
        Calculates Accuracy Ratio on the Train set given cutoff value
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
                        
        # =============================================================================
        # Calculating accuracy      
        # =============================================================================
        
        accuracy = accuracy_score(y_true = self.y_train
                                  , y_pred = self.PredictLabel_Train(cutoff)
                                  )
        
        accuracy = float("{:.3f}".format(accuracy))
        
        return accuracy
    
    def AUC_Train(self) -> float:
        """
        Calculates AUC on the Train set
        """
        auc_score = roc_auc_score(self.y_train, self.Predict_Proba_Train()[:, 1])
        
        auc_score = float("{:.3f}".format(auc_score))
        return auc_score
    
    def Predict_Proba_Train(self) -> np.ndarray:
        """
        Calculates Probability on the Train set
        """
        return self.sk_model.predict_proba(self.x_train)
    
    def Predict_LogProba_Train(self) -> np.ndarray:
        """
        Calculates Logarithm of probability on the Train Set
        """
        return self.sk_model.predict_log_proba(self.x_train)
    
    def PredictLabel_Train(self
                     , cutoff: float
                     ) -> list:
        """
        Calculates Labels Given Cutoff value
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating labels      
        # =============================================================================

        labels = [1 if prob > cutoff else 0 for prob in self.Predict_Proba_Train()[:, 1]]
        return labels
    
    def Brier_Train(self) -> float:
        """
        Calculates Bries Score on the Train Set
        """
        
        brier = brier_score_loss(y_true = self.y_train
                                 , y_prob = self.Predict_Proba_Train()[:, 1]
                                 )
        brier = float("{:.3f}".format(brier))            
        return brier
    
    def F1_Train(self) -> float:
        """
        Calculates F1 score on the Train Set
        """
        
        f1 = f1_score(self.y_train, self.sk_model.predict(self.x_train))
        f1 = float("{:.3f}".format(f1))            
        return f1
    
    def Recall_Train(self, cutoff: float)-> float:
        """
        Calculates Recall on the Train Set
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating recall      
        # =============================================================================

        recall = recall_score(y_true = self.y_train, y_pred = self.PredictLabel_Train(cutoff))
        recall = float("{:.3f}".format(float(recall)))
        return recall
    
    def Precision_Train(self, cutoff: float)-> float:
        """
        Calculates Precision on the Train Set
        """
        
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating precision      
        # =============================================================================
        precision = precision_score(y_true = self.y_train, y_pred = self.PredictLabel_Train(cutoff))
        precision = float("{:.3f}".format(precision))        
        return precision
    
    def Confusion_Matrix_Train(self, cutoff: float) -> np.ndarray:
        """
        Calculates Confusion Matrix of Train Set
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating confusion matrix      
        # =============================================================================
        cm = confusion_matrix(self.y_train, self.PredictLabel_Train(cutoff))    
        return cm
    
    def FPR_Train(self) -> np.ndarray:
        """
        Calculates False Positive Ratio on the Train Set
        """
        fpr = roc_curve(self.y_train,  self.Predict_LogProba_Train()[:, 1])[0]
        # fpr = float("{:.3f}".format(fpr))
        
        return fpr
    
    def TPR_Train(self) -> np.ndarray:
        """
        Calculates True Positive Ratio on the Train Set
        """
        trp = roc_curve(self.y_train,  self.Predict_Proba_Train()[:,1])[1]
        # trp = float("{:.3f}".format(trp))
        
        return trp
    
    def ROC_Curve_Train(self):
        """
        Plots the ROC-curve of the Train Set
        """
        
        fpr, tpr, _ = roc_curve(self.y_train, self.Predict_Proba_Train()[:, 1])
        roc_auc = auc(fpr, tpr)
       
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        return plt.show()
    
    # =============================================================================
    # Test Sample Matrics        
    # =============================================================================
    def Gini_Test(self)-> float:
        """
        Calculates Gini on the Test Set
        """
        
        gini = 2 * self.AUC_Test() - 1
        gini = float("{:.3f}".format(gini))
        return gini
       
    def Accuracy_Test(self
                      , cutoff: float
                      )-> float:
        """
        Calculates Accuracy Ratio on the Test Set Given Cutoff value
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating accuracy      
        # =============================================================================

        accuracy = accuracy_score(
                                    y_true = self.y_test
                                  , y_pred = self.PredictLabel_Test(cutoff)
                                  )
        accuracy = float("{:.3f}".format(accuracy))
        return accuracy

    def AUC_Test(self)-> float:
        """
        Calculates AUC on the Test Set
        """
        auc_score = roc_auc_score(self.y_test, self.Predict_Proba_Test()[:, 1])
        
        auc_score = float("{:.3f}".format(auc_score))
        return auc_score
    
    def Predict_Proba_Test(self)  -> np.ndarray:
        """
        Calculates the Probabilities on the Test Set
        """
        return self.sk_model.predict_proba(self.x_test)
       
    def Predict_LogProba_Test(self) -> np.ndarray:
        """
        Calculates the Logarithm of Probabilities on the Test Set
        """
        return self.sk_model.predict_log_proba(self.x_test)
       
    def PredictLabel_Test(self
                     , cutoff: float
                     ) -> list:
        """
        Calcultes the Labels on the Test Set Given Cutoff Value
        """
        
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating labels      
        # =============================================================================
        labels = [1 if prob > cutoff else 0 for prob in self.Predict_Proba_Test()[:, 1]]
        
        return labels
    
    def Brier_Test(self)-> float:
        """
        Calculates the Brier Score on the Test Set
        """
        brier = brier_score_loss(
                                   y_true = self.y_test
                                 , y_prob = self.Predict_Proba_Test()[:, 1]
                                 )
        brier = float("{:.3f}".format(brier))
        
        return brier

    def F1_Test(self)-> float:
        """
        Calculates F1 Score on the Test set
        """
        f1 = f1_score(self.y_test, self.sk_model.predict(self.x_test))
        f1 = float("{:.3f}".format(f1))        
        return f1
    
    def Recall_Test(self, cutoff: float)-> float:
        """
        Calculates the Recall on the Test Set
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating recall      
        # =============================================================================
        recall = recall_score(y_true = self.y_test, y_pred = self.PredictLabel_Test(cutoff))
        recall = float("{:.3f}".format(float(recall)))
        
        return recall
    
    def Precision_Test(self, cutoff: float)-> float:
        """
        Calculates the Precision on the Test Set
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating Precision      
        # =============================================================================
        precision = precision_score(y_true = self.y_test, y_pred = self.PredictLabel_Test(cutoff))
        precision = float("{:.3f}".format(precision))        

        return precision
    
    def Confusion_Matrix_Test(self, cutoff: float) -> np.ndarray:
        """
        Calculates The Confusion Matrix on the Test Set
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(cutoff, float) or not 0 < cutoff < 1:
            raise ValueError("""The 'cutoff' parameter must be float from 0 to 1; got {}""".format(cutoff))
            
        # =============================================================================
        # Calculating confusion matrix      
        # =============================================================================
        cm = confusion_matrix(self.y_test, self.PredictLabel_Test(cutoff))    
        return cm
    
    def FPR_Test(self) -> np.ndarray:
        """
        Calculates False Positive Ratio on the Test Set
        """
        fpr = roc_curve(self.y_test,  self.Predict_LogProba_Test()[:, 1])[0]
        # fpr = float("{:.3f}".format(fpr))
        
        return fpr
    
    def TPR_Test(self) -> np.ndarray:
        """
        Calculates True Positive Ratio on the Test Set
        """
        trp = roc_curve(self.y_test,  self.Predict_Proba_Test()[:,1])[1]
        # trp = float("{:.3f}".format(trp))
        
        return trp
    
    def ROC_Curve_Test(self):
        """
        Plots ROC Curve on the Test Set
        """
        fpr, tpr, _ = roc_curve(self.y_test, self.Predict_Proba_Test()[:, 1])
        roc_auc = auc(fpr, tpr)
       
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # =============================================================================
    # Model Results        
    # =============================================================================
    def GetCoefficients_SK(self):
        """
        Returns an Array of sklearn Model's Coefficients
        """
        return self.sk_model.coef_
    
    def GetIntercept_SK(self) -> np.ndarray:
        """
        Returns an Intercept of sklearn Model 
        """
        return self.sk_model.intercept_
        
    def Summary(self):
        """
        Returns a Summary of the Model
        """
        return self.sm_model.summary()
    
    def GetCoefficients_SM(self) -> pd.DataFrame:
        """
        Returns an array of statsmodel's coefficients
        """
       
        coefficients = self.sm_model.params.values
        p_value = self.sm_model.pvalues
        variables = self.sm_model.params.index.tolist()
        
        
        final_data = pd.DataFrame(
                {
                    'variable': variables
                    , "coefficient": coefficients
                    , "p_value": [float("{:.4f}".format(i)) for i in p_value]
                }
            )
        return final_data
 

    # =============================================================================
    # Model Predictions on external data     
    # =============================================================================
    def Prediction(self
                   , x_data: pd.DataFrame
                   , logprob: bool = False
                   ) -> pd.Series:
        """
        The method makes prediction based on the external dataset
        """
        # =============================================================================
        # validating input        
        # =============================================================================
        if not isinstance(x_data, pd.DataFrame):
            raise TypeError("""The 'x_data' parameter must be pd.DataFrame""")
      
        # if not x_data.columns.equals(self.x_test.columns):
        #     raise Exception("""Columns of x_data and x_test are not identical""")      
      
        if not isinstance(logprob, bool):
            raise ValueError("""The 'logprob' parameter must be logical""")            
        # =============================================================================
        # Calculating prediction      
        # =============================================================================

        x_data = x_data[self.x_test.columns.tolist()]
        
        if logprob:
            pred = self.sk_model.predict_log_proba(x_data)[:, 1]
        else:
            pred = self.sk_model.predict_proba(x_data)[:, 1]
            
        pred = pd.Series(pred)
        pred.index = x_data.index
        
        return pred
    
    
    




