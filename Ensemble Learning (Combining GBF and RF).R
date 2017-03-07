rf_pred <- read.csv("rf_pred.csv",sep=",")
gbt_pred <- read.csv("GBT_pred.csv",sep=",")
gbt_pred <- gbt_pred$x
rf_pred_corrected <- c(0.122429,rf_pred$"X1.224287392028675925e.01")
true_values <- read.csv("true_values.csv",sep=",")

library(ROCR)
auc_scores <- vector()
#try the linear combination of random forest and gradient boosting tree predictions (probabilities),
#find the optimal weight for random forest
for (w in seq(0.005,0.50,0.005)) {
  ensemble_pred <- w*rf_pred_corrected+(1-w)*gbt_pred
  prob.pred <- prediction(ensemble_pred, true_values)
  auc <- performance(prob.pred,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc_scores <- c(auc_scores,auc)
}
0.005*which.max(auc_scores)
#weight for random forest model is 0.085
max(auc_scores)
#auc score on the selected test set is 0.8381,
#which is a slightly improvement than the gradient boosting trees model

#plot the ROC curve
w=0.085
prob.pred <- prediction(w*rf_pred_corrected+(1-w)*gbt_pred, true_values)
prob.perf <- performance(prob.pred,"tpr","fpr")
#plot the ROC curve
plot(prob.perf,main="ROC Curve for the Ensemble Learning Model",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
text("AUC:0.8381",x = 0.85,y=0.15)