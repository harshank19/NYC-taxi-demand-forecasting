import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"Results\Master_summary_table.csv")

zones = [str(i) for i in df["zone_id"].values]
R2 = df["R2_test"].values

'''
plt.scatter(df["Coefficient_of_variation"], df["R2_test"])
m, b = np.polyfit(df["Coefficient_of_variation"], df["R2_test"], deg=1)
plt.axline(xy1=(0, b), slope=m, color='r', linestyle='--')#, label=f'$y = {m:.4f}x {b:+.4f}$')
plt.title("Coefficient of variation vs R2 Score (test)")
plt.xlabel("Coefficient of variation (CV)")
plt.ylabel("R2 Score")
plt.xlim(50,110)
#plt.legend()
#plt.savefig("CoV_R2.png", dpi=300)
plt.show()

plt.scatter(df["mean_y_test"], df["Relative_MAE"])
m, b = np.polyfit(df["mean_y_test"], df["Relative_MAE"], deg=1)
plt.axline(xy1=(0, b), slope=m, color='r', linestyle='--', label='Regression Line')#, label=f'$y = {m:.4f}x {b:+.4f}$')
plt.title("Relative Mean Absolute Error vs Mean Actual Demand")
plt.xlabel("Mean Actual Demand")
plt.ylabel("Relative MAE")
plt.xlim(60,250)
#plt.legend()
#plt.savefig("Relative_MAE_Mean_Demand.png", dpi=300)
plt.show()


plt.scatter(df["R2_train"], df["R2_test"])
m, b = np.polyfit(df["R2_train"], df["R2_test"], deg=1)
plt.axline(xy1=(0, b), slope=m, color='r', linestyle='--')#, label=f'$y = {m:.4f}x {b:+.4f}$')
plt.title("Overfitting Check")
plt.xlabel("R2 Score (Train)")
plt.ylabel("R2 Score (Test)")
plt.xlim(0.975, 1)
plt.ylim(0.785, 1)
#plt.legend()
#plt.savefig("Overfitting_Check.png", dpi=300)
plt.show()


df_bar = pd.DataFrame({"zones": zones, "R2_test": R2})
df_bar = df_bar.sort_values(by="R2_test", ascending=False)

plt.bar(df_bar["zones"], df_bar["R2_test"])
plt.xlabel("Zone ID")
plt.ylabel("R2 Score")
plt.title("R2 Score for Selected Zones")
#plt.savefig("Zone_wise_R2_Score.png", dpi=300)
plt.show()
'''
'''
worst_R2_zone = df.loc[df["R2_test"] == df["R2_test"].min(),"zone_id"].values[0]
best_R2_zone = df.loc[df["R2_test"] == df["R2_test"].max(),"zone_id"].values[0]
median_R2_zone = df.loc[df["R2_test"] == df["R2_test"].median(),"zone_id"].values[0]

df2 = pd.read_csv(r"Results\all_zone_test_predictions.csv")

worst_R2_df = df2[df2["zone_id"] == worst_R2_zone]
worst_R2_residuals = (worst_R2_df["predicted"] - worst_R2_df["actual"]).values

best_R2_df = df2[df2["zone_id"] == best_R2_zone]
best_R2_residuals = (best_R2_df["predicted"] - best_R2_df["actual"]).values

median_R2_df = df2[df2["zone_id"] == median_R2_zone]
median_R2_residuals = (median_R2_df["predicted"] - median_R2_df["actual"]).values


bin = np.linspace(-1.25, 1.25, 50)
#bin=30
plt.hist(worst_R2_residuals/worst_R2_df["actual"].mean(), color="red", bins=bin, alpha=0.6, histtype='step', label=f"Worst Performing Zone (ID: {worst_R2_zone})")
plt.hist(median_R2_residuals/median_R2_df["actual"].mean(), color='blue', bins=bin, alpha=0.6, histtype='step', label=f"Median Performing Zone (ID: {median_R2_zone})")
plt.hist(best_R2_residuals/best_R2_df["actual"].mean(), color='green', bins=bin, alpha=0.6, histtype='step', label=f"Best Performing Zone (ID: {best_R2_zone})")
plt.xlabel("Residual / Mean demand")
plt.title("Standardized Residuals for Worst-Median-Best Performing Zones")
plt.legend()
#plt.savefig("Standardized_Residuals.png", dpi=300)
plt.show()


plt.scatter(worst_R2_df["actual"], worst_R2_residuals/worst_R2_df["actual"].std(), label=f"Worst Performing Zone (ID: {worst_R2_zone})", marker='s', facecolors='none', edgecolors='red',alpha=0.7)
plt.scatter(median_R2_df["actual"], median_R2_residuals/median_R2_df["actual"].std(), label=f"Median Performing Zone (ID: {median_R2_zone})", marker='*', facecolors='none', edgecolors='blue',alpha=0.5)
plt.scatter(best_R2_df["actual"], best_R2_residuals/best_R2_df["actual"].std(), label=f"Best Performing Zone (ID: {best_R2_zone})", marker='^', facecolors='none', edgecolors='green',alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Demand")
plt.ylabel("Standardised Residual")
plt.title("Standardised Residual vs Predictions for Worst-Median-Best Performing Zones")
plt.legend()
#plt.savefig("Predicted_vs_Residuals.png", dpi=300)
plt.show()
'''
'''
## Feature Importance
feature_importance_df = pd.read_csv(r"Results\feature_importance_by_zone.csv")
feature_importance_df = feature_importance_df.drop(columns="zone_id")
feature_names = feature_importance_df.columns
feature_importances = [feature_importance_df[i].mean() for i in feature_names]

importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(15,6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Average Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()
'''