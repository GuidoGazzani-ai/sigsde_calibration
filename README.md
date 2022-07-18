# sigsde_calibration

This is a collection of Jupyter notebooks and Python files which have been used in the article "Signature-based models: theory and calibration" of Christa Cuchiero, Guido Gazzani and Sara Svaluto-Ferro. (add link)
In the present repository you will find the following material.
<div class="about">
                <h2 style="color:#06386D"><b>Calibration to time-series data</b></h2>
  <ul>
<li>Code for a Heston model, when learning the price dynamics. (Stoch_vol_regressionPrice_Heston.py)</li><br>
<li>Code for a SABR-type model, when learning the volatility of the price dynamics. (Stoch_vol_regressionQV_SABR.py)</li><br>
  </ul>
  </div>
  Details of the calibration to time-series data with signature-based model can be found in Section 4.1 of the paper.
  
  <div class="about">
                <h2 style="color:#06386D"><b>Calibration to option prices</b></h2>
  <ul>
<li>Code for a Heston model generated implied volatility surface with constant model parameters.(MC_Heston.ipynb)</li><br>
    <li> Code for market-data with constant parameters.(MC_market_calibration.ipynb)</li><br>
<li>Code for a market-data with time-varying parameters. (Cluster_MC_Time_Varying.py ran on the UniWien HPC3 Cluster and the corresponding notebook to visualize the results Calibration_TimeVarying.ipynb)</li><br>
  </ul>
  </div>
  Details of the calibration to option prices with signature-based model can be found in Section 4.2 of the paper.
    <div class="about">
                <h2 style="color:#06386D"><b>Joint calibration to time-series data and option prices</b></h2>
  <ul>
<li>Code for a Heston model with constant parameters. (Joint_Calibration.py)</li><br>
  </ul>
  </div>
  A brief discussion concerning the joint calibration problem can be found in Section 4.3 of the paper.
  
  <br>
<br>
  <br>
  <br>
    <br>
  <br>
    
    
![multi_dimensional_BS](https://user-images.githubusercontent.com/58938961/164503914-29352ed2-69f8-4d7a-97c4-5847b32d7140.png)
