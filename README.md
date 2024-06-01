# Is sympathy play a viable strategy?
## Team: Devadatta Hegde, Guanqian Wang, JinJin Zhang, Kang Lu, Keshav Sutrave, Nathaniel Tamminga


### Problem

The stock prices of companies in the same sector appear to respond to events with a direct correlation. The goal of this project is to build models to analyze whether this feature could be exploited to advantage in the markets. 

**Stakeholders**: Private Investors, Investment firms and Hedge funds.

**Key Performance Indicators (KPI)**:
1. Having some insights about stock price data within each sector. (data pre-processing and EDA)
2. Having a pool of candidates of models to investigate. Those models should be reasonable to analyze stock market data.
3. The most ideal goal is to have a model that indicates that sympathy play as a factor would be statistically significant or improve predictability. If not, we would at least be able to conclude that under our assumptions and data we can acquire, sympathy play is not valid.
4. We will provide interpretations of our results.

**Data sets**: We will build the model using the stock prices of a few companies in the following sectors.

1. Retail
2. Banks
3. Semiconductor
4. Internet
5. Pharmaceuticals and healthcare.
6. Airlines

We will scrape daily stock price data from `Yahoo Finance`.

**Modeling Approach**:
See the attached file named Proposed model approach.

**Repository Organization**
This repo is organized as follows:
- data: subdirectory containing the data used for this project
- EDA_demo: subdirectory containing the python scripts used to acquire and format the data for analysis
- SectorStockList: subdirectory containing the text files listing the stocks we will look at in each sector
- SectorSummaries: subdirectory containing the summaries of exploring sympathy play in each of our respective stock sectors
- Workflow_demo: subdirectory containing the python scripts of our different forecasting models. These scripts were used to generate the plots used in our analysis of sympathy play.
- Proposed_modeling_approach: This file comes in both a word document and a markdown file. These files explain our approach to modelling and analyzing sympathy play as a valid investment strategy.
- Executive Summary: This pdf file contains a summary combining our analysis for each of the different sectors.