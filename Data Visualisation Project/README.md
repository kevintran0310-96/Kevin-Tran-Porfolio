# Youth Offender Analysis Dashboard 🇦🇺

This project delivers an interactive R Shiny dashboard for comprehensive analysis of youth offending rates across Australia, exploring their relationship with screen usage and various demographic factors. The dashboard offers dynamic visualizations to uncover trends, geographical disparities, gender-based differences, and potential correlations, providing actionable insights for policymakers and researchers.

## 🌟 Project Overview

Youth offending is a complex societal issue influenced by numerous factors. This project aims to shed light on its dynamics by integrating diverse datasets to answer critical questions:

* **When** do youth crimes occur?

* **Where** are youth crimes most prevalent?

* **Why** might youth crimes occur, considering factors like screen usage and demographics?

* **Who** is primarily involved in youth crime?

By leveraging data from the Australian Bureau of Statistics (ABS) and other governmental sources, this dashboard serves as a powerful tool for understanding youth behavior, identifying at-risk populations, and informing evidence-based interventions.

## ✨ Features & Visualizations

The dashboard is structured into several interactive sections, each offering unique insights:

1. **Youth Offender Offences Over Time**:

   * **Interactive Heatmap**: Visualizes the geographical distribution of youth offenses across Australian states, highlighting areas with higher or lower crime rates.

   * **Animated Line Chart**: Displays temporal trends in youth offenses for a selected state (or overall if no state is selected), showing year-on-year changes.

   * **Insight**: Reveals state-specific and national trends, often linked to socio-economic events, and identifies shifts in offence types (e.g., rise in digital-related offences).

2. **Gender Analysis of Youth Offenders**:

   * **Interactive Bubble Chart**: Explores gender dynamics of youth offending by state, age group, and year.

   * **Gender Toggles**: Allows users to switch between male and female offender statistics.

   * **Insight**: Highlights significant disparities, with males consistently outnumbering females, particularly in the 14-17 age group, and regional differences in gender gaps.

3. **Screen Usage Purposes Across Australia**:

   * **Interactive Heatmap**: Depicts children's screen usage purposes (e.g., calling family, internet access, gaming) across different states over time.

   * **Animated Year Slider**: Allows dynamic exploration of screen usage patterns year by year.

   * **Insight**: Identifies popular screen-based activities and variations in digital engagement across states, potentially indicating local socio-economic or cultural influences.

4. **Correlation between Screen Usage and Youth Offending**:

   * **Interactive Scatter Plot**: Analyzes the relationship between the number of children using electronic devices and youth offending rates. Includes a trend line and predicted values.

   * **Venn Diagram**: Visualizes the shared variance between screen usage and youth offending rates, quantifying the overlap.

   * **Insight**: Uncovers a strong positive correlation (e.g., Pearson correlation coefficient of 0.94) suggesting a link to increased screen time and youth crime, while emphasizing that other factors also play a significant role.

## 📊 Data Sources

The reliability and comprehensiveness of this project are built upon data from reputable Australian sources:

* **Youth Offenders Data**: Sourced from the [Australian Bureau of Statistics (ABS) - Recorded Crime Offenders](https://www.abs.gov.au/statistics/people/crime-and-justice/recorded-crime-offenders/latest-release#data-downloads) (2013-2023).

* **Participation in Cultural Activities Data (Screen Use)**: Obtained from [ABS - Participation in Selected Cultural Activities](https://www.abs.gov.au/statistics/people/people-and-communities/participation-selected-cultural-activities/latest-release).

* **Geographic Data (State Boundaries)**: Shapefiles from the [Australian Bureau of Statistics (ABS) - Australian Statistical Geography Standard (ASGS)](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files).

* **Population Data**: `YO Sex and Age by States.csv` (used for gender analysis).

* **Specific Year Data for Correlation**: `Screen by State 2017.csv` and `Youth Offender Rate in 2017.csv`.

*(Note: Ensure `data/` folder contains these CSV files and the shapefile, or provide clear download instructions if data isn't included in the repo.)*

## 🛠️ Technologies & Libraries

This project is built using the R programming language and heavily relies on the following R packages:

* `shiny`: For creating interactive web applications.

* `leaflet`: For interactive mapping and geographical visualizations.

* `dplyr`: For efficient data manipulation and transformation.

* `sf`: For handling spatial data (shapefiles).

* `ggplot2`: For static and animated data visualizations.

* `gganimate`: To create dynamic, animated `ggplot2` plots.

* `plotly`: For converting `ggplot2` plots into interactive web graphics.

* `gifski`: For rendering `gganimate` animations to GIF format.

* `reshape2`: For reshaping data, particularly for heatmap generation.

* `GGally`: For creating matrix plots (though not directly used in the final deployed charts, it's a dependency).

* `gridExtra`, `VennDiagram`, `grid`: For arranging multiple plots and creating Venn diagrams.

* `shinythemes`: For styling the Shiny application with pre-defined themes.

## 🚀 How to Run the Dashboard

To run this R Shiny dashboard locally:

1. **Clone the Repository**:

   ```bash
   git clone [https://github.com/kevintran0310-96/Kevin-Tran-Porfolio.git](https://github.com/kevintran0310-96/Kevin-Tran-Porfolio.git)
   cd Kevin-Tran-Porfolio/Youth_Offender_Dashboard
   ```

2. **Install R Packages:**:

If you don't have them already, install the required R packages from within your R environment:
   ```bash
   install.packages(c("shiny", "leaflet", "dplyr", "sf", "ggplot2", "gganimate", "plotly",
                   "gifski", "reshape2", "GGally", "gridExtra", "VennDiagram", "grid", "shinythemes"))
```
3. **Place Data Files:**:

Ensure the following data files are present in the Youth_Offender_Dashboard/data/ directory (or the root directory of this project, depending on your app.R's file paths):

* Cleaned YO by Years.csv

* YO Sex and Age by States.csv

* Screen Purpose.csv

* Screen by State 2017.csv

* Youth Offender Rate in 2017.csv

* STE_2021_AUST_GDA2020.shp (and associated shapefile components like .dbf, .shx, .prj)

Self-correction note: The app.R loads files directly from the working directory, so it's best to place them in the same folder as app.R or update paths in app.R if you create a data/ folder.

4. **Run the App:**:

Open RStudio, navigate to the Youth_Offender_Dashboard directory, and run the application:

```bash
shiny::runApp()
```

Alternatively, you can open the app.R file in RStudio and click the "Run App" button.

## 💡 Key Insights & Conclusion
The dashboard's analyses reveal critical insights into youth offending in Australia:

Geographical Disparities: States like New South Wales and Queensland consistently show higher concentrations of youth offenders, with observable yearly fluctuations.

Gender and Age Profile: Youth crime is most prevalent among males aged 14-17, suggesting targeted interventions for at-risk male teenagers in high-crime states could be beneficial.

Digital Engagement Patterns: While core screen activities like communication and internet access are widespread, variations across states align with differing crime rates. This hints at digital habits potentially influencing vulnerable youth.

Correlation between Screen Use and Crime: A significant positive correlation between screen usage and youth crime exists. However, it is crucial to understand that correlation does not imply causation. Screen time is one of many factors, alongside socio-economic conditions, contributing to youth behavior.

This project underscores the multifaceted nature of youth crime. Addressing it effectively requires a holistic approach that considers digital habits alongside other socio-economic drivers. Further longitudinal studies would be invaluable in untangling these complex relationships to develop more effective, evidence-based solutions.

## 📧 Contact
Feel free to connect or reach out if you have any questions or collaboration opportunities:

LinkedIn: [https://www.linkedin.com/in/kevintran0310/]

Email: [khoatran031096@gmail.com]

📄 License
This project is open-sourced under the MIT License.
