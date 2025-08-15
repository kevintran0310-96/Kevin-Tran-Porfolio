# app.R

# ---- Packages ----
library(shiny)
library(shinythemes)
library(dplyr)
library(ggplot2)
library(plotly)
library(sf)
library(leaflet)
library(eulerr) 
library(grid)
library(gridExtra)

# ---- Helpers ----
# Prefer files in ./data if present; fall back to working dir.
path_for <- function(fname) {
  fp <- file.path("data", fname)
  if (file.exists(fp)) fp else fname
}

# ---- Data loads ----
# Core aggregated offences data
data <- read.csv(path_for("Cleaned YO by Years.csv"), stringsAsFactors = FALSE, check.names = TRUE)
# Column names in your file: State, Youth Offender Type, Year, Offences
# read.csv turns "Youth Offender Type" -> Youth.Offender.Type
data$Year <- suppressWarnings(as.integer(data$Year))
data$Offences <- suppressWarnings(as.numeric(gsub(",", "", data$Offences)))
data$Youth.Offender.Type <- as.factor(data$Youth.Offender.Type)
data <- data |> filter(!is.na(Year), !is.na(Youth.Offender.Type), !is.na(Offences))

# Shapefile (needs .shp, .shx, .dbf, .prj together)
shp_try <- path_for("STE_2021_AUST_GDA2020.shp")
states_shape <- tryCatch({
  if (file.exists(shp_try)) st_read(shp_try, quiet = TRUE) |> st_transform(4326) else NULL
}, error = function(e) NULL)

# Population by sex/age/state (columns: Year, Sex, Age, State=abbr, Population)
population_data <- read.csv(path_for("YO Sex and Age by States.csv"),
                            stringsAsFactors = FALSE, check.names = TRUE)

# Screen purpose data (columns: Purpose, Percentage(with %), State, Year)
data_screen_purpose <- read.csv(path_for("Screen Purpose.csv"),
                                stringsAsFactors = FALSE, check.names = TRUE)
data_screen_purpose$Percentage <- as.numeric(sub("%", "", data_screen_purpose$Percentage))

# Correlation datasets (2017)
screen_data <- read.csv(path_for("Screen by State 2017.csv"),
                        stringsAsFactors = FALSE, check.names = TRUE)
youth_offender_data <- read.csv(path_for("Youth Offender Rate in 2017.csv"),
                                stringsAsFactors = FALSE, check.names = TRUE)

# ---- Clean & merge for correlation section ----

# 1) Screen-by-state: rename long header -> Device_Users; trim State
if (!"Device_Users" %in% names(screen_data)) {
  long_nm <- "Number.of.Children.Participating.in.Screen.based.Acitivities"
  if (long_nm %in% names(screen_data)) {
    names(screen_data)[names(screen_data) == long_nm] <- "Device_Users"
  }
}
if ("State" %in% names(screen_data)) {
  screen_data$State <- trimws(screen_data$State)
}
screen_data$Device_Users <- suppressWarnings(as.numeric(gsub(",", "", screen_data$Device_Users)))

# 2) Youth offender numbers:
if (!"State" %in% names(youth_offender_data) && "Year" %in% names(youth_offender_data)) {
  if (all(!grepl("^[0-9]+$", youth_offender_data$Year))) {
    names(youth_offender_data)[names(youth_offender_data) == "Year"] <- "State"
  }
}
ynm <- "Youth Offender Number"
if (ynm %in% names(youth_offender_data)) {
  names(youth_offender_data)[names(youth_offender_data) == ynm] <- "Youth.Offender.Number"
}
if ("Youth.Offender.Number" %in% names(youth_offender_data)) {
  youth_offender_data$Youth.Offender.Number <- as.numeric(gsub(",", "", youth_offender_data$Youth.Offender.Number))
}

# 3) Merge & model
if (!all(c("State","Device_Users") %in% names(screen_data)) ||
    !all(c("State","Youth.Offender.Number") %in% names(youth_offender_data))) {
  merged_data <- data.frame(State = character(), Device_Users = numeric(), Youth_Offenders = numeric())
} else {
  merged_data <- merge(
    screen_data[, c("State","Device_Users")],
    youth_offender_data[, c("State","Youth.Offender.Number")],
    by = "State", all = TRUE
  )
  names(merged_data)[names(merged_data) == "Youth.Offender.Number"] <- "Youth_Offenders"
  merged_data <- merged_data[stats::complete.cases(merged_data), ]
}
lm_fit <- if (nrow(merged_data) >= 2) lm(Youth_Offenders ~ Device_Users, data = merged_data) else NULL

# ---- UI ----
intro_ui <- tagList(
  fluidPage(
    fluidRow(
      column(12, 
             h2("Project Overview"),
             p("This project aims to provide a comprehensive analysis of youth offending rates in Australia by examining their relationship with screen usage and other demographic factors. The dashboard utilizes multiple datasets to explore key questions about when, where, and why youth crimes occur, as well as the potential influence of digital engagement on youth behavior."),
             p("The study leverages data from various governmental and statistical sources, including the Australian Bureau of Statistics, to provide insights into geographical trends, gender-based differences, and behavioral patterns. By combining these data sources, the project aims to deliver meaningful and actionable insights to help policymakers, researchers, and the community better understand the dynamics of youth crime."),
             h3("Visualizations"),
             p("This dashboard presents several interactive visualizations to illustrate trends and correlations across different dimensions of youth crime and behavior:"),
             tags$ul(
               tags$li(HTML("<b>Heatmap and Line Chart:</b> These visualizations provide a geographical and temporal overview of youth offences across Australia. Users can select a specific year and offender type to explore how youth crime is distributed across different states, and click on a state to observe a detailed trend over time.")),
               tags$li(HTML("<b>Gender Analysis - Male/Female Population:</b> This section allows users to explore the gender dynamics of youth offending. Using buttons and dropdown menus, users can select a gender, state, and year to visualize the distribution of offences by age group, providing insights into which gender and age categories are most affected.")),
               tags$li(HTML("<b>Screen Usage Heatmap:</b> This heatmap depicts children's screen usage purposes across different states, offering insights into the types of screen-based activities (such as calling family, accessing the internet, or playing games) that are popular across different regions.")),
               tags$li(HTML("<b>Correlation Analysis - Screen Usage and Youth Offending:</b> This section uses a scatter plot and Venn diagram to analyze the correlation between screen usage and youth crime rates. Users can input the number of children using electronic devices to predict youth offending numbers, and observe the relationship between the two variables."))
             ),
             h3("How to Use the Visualizations"),
             p("When first opening Visualisations and Insight tabs, please wait around 10 seconds for the data to load. The visualizations included in this app are designed to provide a clear and interactive way to explore youth offending behaviors, screen usage, and demographic insights. Below are detailed instructions for using each visualization to gain maximum insights:"),
             tags$ul(
               tags$li(HTML("<b>Heatmap and Line Chart:</b> 
          Use the slider to select a specific <b>year</b> and filter by <b>offender type</b> to visualize the youth crime distribution across states.
          <ul>
            <li>The <b>heatmap</b> shows the intensity of youth offences by state, with color gradients indicating the frequency of offences. Click on different states to identify regions with higher or lower offending rates.</li>
            <li>Clicking on a specific state will also update the <b>animated line chart</b> on the right, which then displays the trend in youth offences for that state over time. This allows you to track how crime rates have changed year by year.</li>
          </ul>")),
               tags$li(HTML("<b>Gender Analysis - Male/Female Population Analysis:</b> 
          This section allows you to explore gender-based differences in youth offending across states.
          <ul>
            <li>Use the <b>Male</b> and <b>Female</b> buttons to toggle between different gender perspectives, and observe the changes in crime data accordingly.</li>
            <li>Select a specific <b>state</b> and <b>year</b> from the dropdown menus to view the gender-based offender statistics for that state. The bubble chart will illustrate the <b>age distribution</b> of youth offenders for the selected criteria, highlighting which age groups are most involved in offences.</li>
            <li>The size of the bubbles indicates the number of offenders in each age group, providing a visual comparison between different groups and genders.</li>
          </ul>")),
               tags$li(HTML("<b>Screen Usage Heatmap:</b> 
          Explore how children's screen usage varies by state and purpose over different years.
          <ul>
            <li>Use the <b>year slider</b> to select a specific year, and the heatmap will show how children in different states use electronic devices for various purposes like <b>communication</b>, <b>education</b>, and <b>entertainment</b>.</li>
            <li>The colors on the heatmap indicate the percentage of children engaging in each type of activity. Darker colors represent higher percentages, making it easy to identify which activities dominate in different regions.</li>
            <li>Hover over each cell in the heatmap to see the exact percentage value for that combination of state and screen activity, providing more precise data points for analysis.</li>
          </ul>")),
               tags$li(HTML("<b>Correlation Analysis - Screen Usage and Youth Offending:</b> 
          This section provides insights into the correlation between the number of screen users and youth offenders, helping to identify potential relationships between digital engagement and offending behaviors.
          <ul>
            <li>Enter the <b>number of children using electronic devices</b> in the input box, and click the <b>Predict Youth Offenders</b> button to estimate the expected number of youth offenders based on the model.</li>
            <li>The <b>scatter plot</b> shows the data points for each state, plotting the number of screen users against youth offenders. A <b>trend line</b> is also included to indicate the overall correlation direction and strength.</li>
            <li>The <b>Venn diagram</b> below visualizes the shared variance between screen users and youth offenders, highlighting the percentage of influence one factor may have over the other.</li>
            <li>Hover over data points on the scatter plot to view more detailed information, such as the state name and the exact values of screen users and youth offenders.</li>
          </ul>"))
             ),
             h3("Data Sources"),
             p("The data used in this project comes from various authoritative sources, ensuring reliability and comprehensiveness:"),
             tags$ul(
               tags$li(HTML("<b>Youth Offenders Data:</b> Sourced from the Australian Bureau of Statistics (ABS), this dataset provides information on youth crime from 2013 to 2023. Available at: <a href='https://www.abs.gov.au/statistics/people/crime-and-justice/recorded-crime-offenders/latest-release#data-downloads'>ABS Website</a>")),
               tags$li(HTML("<b>Participation in Cultural Activities Data:</b> This dataset includes insights on children's participation in screen-based activities, obtained through national surveys. Available at: <a href='https://www.abs.gov.au/statistics/people/people-and-communities/participation-selected-cultural-activities/latest-release'>ABS Statistics</a>")),
               tags$li(HTML("<b>Geographic Data:</b> Shapefiles used for spatial analysis are sourced from the Australian Bureau of Statistics for defining state boundaries. Available at: <a href='https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files'>ABS Geospatial Data</a>"))
             )
      )
    )
  )
)

viz_ui <- tagList(
  fluidPage(
    fluidRow(
      column(12,
             h2("Section 1: Youth Offender Offences Over Time"),
             p("In the first section, we analyze the geographical distribution of youth offences as well as the temporal trends in these offences using a heatmap and an animated line chart. An additional aspect explored here is the breakdown of offence types, which helps to understand not just the quantity of youth crime but also its nature."),
             p(HTML("<b>Key Insights:</b> The heatmap shows substantial variation across states, with <b>NSW and Victoria</b> showing consistently higher youth crime rates compared to other states like <b>Tasmania</b> or the <b>Northern Territory</b>. The animated line chart reveals fluctuations in offence rates over the years, with notable peaks, possibly linked to socio-economic events. <br>
               Breaking down offence types, there is a <b>decline in serious crimes</b> such as homicide in NSW, but a <b>rise in minor and digital-related offences</b> nationwide, highlighting a shift towards online crimes among youth."))
      )
    ),
    fluidRow(
      column(4,
             wellPanel(
               sliderInput("year", "Select Year:",
                           min = if (nrow(data)) min(data$Year, na.rm = TRUE) else 2013,
                           max = if (nrow(data)) max(data$Year, na.rm = TRUE) else 2023,
                           value = if (nrow(data)) max(data$Year, na.rm = TRUE) else 2023,
                           step = 1, sep = ""),
               selectInput("offender_type", "Select Offender Type:",
                           choices = if (nrow(data)) levels(data$Youth.Offender.Type) else character(),
                           selected = if (nrow(data)) levels(data$Youth.Offender.Type)[1] else NULL)
             )
      ),
      column(8,
             fluidRow(
               column(6, leafletOutput("map", height = "500px")),
               column(6, plotlyOutput("linePlot", height = "600px"))
             )
      )
    ),
    fluidRow(
      column(12,
             p(HTML("With this foundational understanding of where and when youth crime is most prevalent, we move next to delve into a key demographic—gender—to better understand who is involved in youth crime, and whether gender differences provide further insight into the patterns observed."))
      )
    ),
    hr(),
    fluidRow(
      column(12,
             h2("Section 2: Gender Analysis of Youth Offenders"),
             p("This section uses a gender analysis to examine the dynamics of youth offending across Australian states, with interactive buttons that allow users to filter by gender and explore the data visually."),
             p(HTML("<b>Key Insights:</b> The <b>gender analysis</b> reveals a significant disparity, with <b>males consistently outnumbering females</b> in youth crime across all states and age groups. This is especially prominent in the <b>14-17 age group</b>, pointing to underlying socio-cultural factors influencing male youth behavior. States such as <b>Queensland</b> and <b>Western Australia</b> exhibit larger gender gaps, suggesting that regional differences may contribute to these trends, potentially driven by economic conditions or community environments."))
      )
    ),
    fluidRow(
      column(4, align = "center",
             wellPanel(
               fluidRow(
                 column(6, actionButton("male_button", "Male", icon = icon("male"),
                                        style = "font-size: 20px; color: #FFFFFF; background-color: #007BFF; width: 90%; height: 70px")),
                 column(6, actionButton("female_button", "Female", icon = icon("female"),
                                        style = "font-size: 20px; color: #FFFFFF; background-color: #FF69B4; width: 90%; height: 70px"))
               ),
               br(),
               selectInput("state_input", "Select State:",
                           choices = c("New South Wales","Victoria","Queensland","South Australia",
                                       "Western Australia","Tasmania","Northern Territory","Australian Capital Territory")),
               br(),
               selectInput("year_input", "Select Year:", choices = sort(unique(population_data$Year)))
             )
      ),
      column(8, plotlyOutput("population_plot", height = "600px"))
    ),
    fluidRow(
      column(12,
             p(HTML("Understanding that males, particularly those in their late teenage years, are more likely to be involved in youth crime brings us to the question of potential influencing factors. With the increasing integration of digital technologies in youths' daily lives, we next investigate whether screen-based activities play a role in shaping these behaviors."))
      )
    ),
    hr(),
    fluidRow(
      column(12,
             h2("Section 3: Screen Usage Purposes Across Australia"),
             p("This section presents an interactive heatmap depicting how children in different states use screen-based devices for various purposes, showing data over time and exploring key types of digital engagement."),
             p(HTML("<b>Key Insights:</b> The heatmap shows that activities such as '<b>calling parents/family</b>' and '<b>accessing the internet</b>' are consistently popular across states, reflecting the integral role of digital devices in communication and learning. States like <b>Western Australia (WA)</b> and <b>Queensland (QLD)</b> report higher levels of screen engagement compared to other states, suggesting local factors such as socio-economic conditions or cultural preferences influence these behaviors. Activities like '<b>listening to music</b>' or '<b>receiving calls</b>' are notably lower, indicating some variance in how different activities appeal to youths."))
      )
    ),
    fluidRow(
      column(4,
             wellPanel(
               sliderInput("screen_year", "Select Year:",
                           min = min(data_screen_purpose$Year, na.rm = TRUE),
                           max = max(data_screen_purpose$Year, na.rm = TRUE),
                           value = min(data_screen_purpose$Year, na.rm = TRUE),
                           step = 1, sep = "", animate = animationOptions(interval = 1000))
             )
      ),
      column(8, plotlyOutput("heatmap_screen_usage"))
    ),
    fluidRow(
      column(12,
             p(HTML("The variation in screen-based activities across states points towards the next logical question—could these digital behaviors be influencing youth crime? To explore this further, the next section correlates digital engagement with youth offending rates to see if a link exists between increased screen use and youth crime involvement."))
      )
    ),
    hr(),
    fluidRow(
      column(12,
             h2("Section 4: Correlation between Screen Usage and Youth Offending"),
             p("In this section, we analyze the relationship between the number of children using electronic devices and youth offending rates."),
             p(HTML("<b>Key Insights:</b> The analysis reveals a strong positive correlation (<b>Pearson correlation coefficient of 0.94</b>) between the number of children engaged in screen-based activities and youth offender rates, suggesting that increased screen time is linked to a rise in youth crime. The <b>Venn diagram</b> indicates around <b>25-30%</b> of the variance in youth crime is shared with screen usage, highlighting the significance of digital engagement as a factor. However, the results also stress that other variables—such as socio-economic conditions—contribute significantly, meaning screen usage is one of many influences."))
      )
    ),
    fluidRow(
      column(4,
             wellPanel(
               numericInput("Device_Users_input", "Enter the Number of Children Using Electronics Devices:", value = 100000, min = 0),
               actionButton("predict_button", "Predict Youth Offenders")
             )
      ),
      column(8, plotlyOutput("scatterPlot", height = "600px"))
    ),
    fluidRow(column(12, plotOutput("vennAndTextPlot", height = "400px"))),
    hr(),
    fluidRow(
      column(12,
             h2("Conclusion"),
             p("1. Geographic Disparities: New South Wales and Queensland show the highest concentration of youth offenders, with significant yearly fluctuations that could indicate socio-economic influences."),
             p("2. Gender and Age: Youth crime is most prevalent among 14-17-year-old males. These insights suggest that interventions should be particularly focused on at-risk male teenagers in states with high crime rates."),
             p("3. Digital Engagement Patterns: The majority of children use their devices for communication and accessing the internet. Certain states have higher screen engagement, aligning with higher crime rates, pointing towards digital habits potentially being a risk factor for vulnerable youth."),
             p("4. Correlation Between Screen Use and Crime: The strong positive correlation between screen use and youth crime highlights the potential risk of excessive digital engagement. However, this does not establish causation and calls for a nuanced understanding of the various factors involved."),
             p("Final Thought: Each visualization provides a piece of the puzzle in understanding youth crime in Australia. Moving forward, it is crucial for policymakers and community leaders to consider the multifaceted nature of these findings. Addressing youth crime should involve reducing excessive screen time through education and promoting healthy, supervised digital habits, while also considering other key socio-economic drivers. The journey does not end here—further research, especially longitudinal studies, could help untangle the complex relationships between digital engagement and youth behavior, contributing to more effective, evidence-based solutions.")
      )
    )
  )
)

ui <- navbarPage(
  title = "Youth Offender Analysis Dashboard",
  theme = shinytheme("flatly"),
  id = "top_tabs",
  tabPanel("Introduction", intro_ui),
  tabPanel("Visualizations and Insights", viz_ui)
)

# ---- Server ----
server <- function(input, output, session) {
  
  # Reactive value to store selected sex
  selected_sex <- reactiveVal("Male")
  
  # Update selected sex when male/female button is clicked
  observeEvent(input$male_button, { selected_sex("Male") })
  observeEvent(input$female_button, { selected_sex("Female") })
  
  # Render the plot based on selected sex, state, and year
  output$population_plot <- renderPlotly({
    req(nrow(population_data) > 0)
    
    # Map full state names to abbreviations for consistency
    state_mapping <- c(
      "New South Wales" = "NSW",
      "Victoria" = "Vic",
      "Queensland" = "Qld",
      "South Australia" = "SA",
      "Western Australia" = "WA",
      "Tasmania" = "Tas",
      "Northern Territory" = "NT",
      "Australian Capital Territory" = "ACT"
    )
    
    # Filter data based on selected state, gender, and year
    selected_state_abbreviation <- state_mapping[[input$state_input]]
    gender <- selected_sex()
    
    data_filtered <- population_data %>% 
      filter(State == selected_state_abbreviation & Year == input$year_input & Sex == gender)
    
    validate(need(nrow(data_filtered) > 0, "No data available for the selected filters."))
    
    # Create a ggplot for Bubble Chart
    p <- ggplot(data_filtered, aes(x = Age, y = Population, size = Population, color = Age)) +
      geom_point(alpha = 0.7) +
      scale_size(range = c(5, 20)) +
      labs(title = paste(gender, "Youth Offender Number in", input$state_input, "for Year", input$year_input),
           x = "Age Group",
           y = "Youth Offender Number",
           size = "Youth Offender Number") +
      theme_minimal()
    
    ggplotly(p, tooltip = c("x", "size"))
  })
  
  # Reactive filtered dataset for the latest year and offender type based on user input
  filtered_data <- reactive({
    req(nrow(data) > 0)
    data %>% 
      filter(Year == input$year, Youth.Offender.Type == input$offender_type) %>%
      group_by(State) %>%
      summarise(Total_Offences = sum(Offences, na.rm = TRUE), .groups = "drop")
  })
  
  # Render the leaflet heatmap
  output$map <- renderLeaflet({
    validate(need(!is.null(states_shape), "Map data (shapefile) could not be loaded."))
    plot_data <- filtered_data()
    
    # Find shapefile state-name field and join
    shp_name_col <- intersect(c("STE_NAME21", "STATE_NAME", "STATE_NAME_2021", "STATE_NAME16"), names(states_shape))
    validate(need(length(shp_name_col) >= 1, "State name column not found in shapefile."))
    shp_name_col <- shp_name_col[1]
    
    map_data <- states_shape %>% 
      left_join(plot_data, by = setNames("State", shp_name_col))
    
    # Create a stable column to use in formulas (avoid .data pronoun)
    map_data$SHP_NAME <- map_data[[shp_name_col]]
    
    pal <- colorNumeric("YlOrRd", domain = plot_data$Total_Offences, na.color = "#BDBDBD")
    
    leaflet(map_data) %>% 
      addTiles() %>% 
      addPolygons(
        fillColor = ~pal(Total_Offences),
        fillOpacity = 0.7,
        color = "black",
        weight = 1,
        layerId = ~SHP_NAME,   # concrete column (fixes :: data mask error)
        label = ~SHP_NAME,     # concrete column
        labelOptions = labelOptions(
          style = list("font-weight" = "normal", padding = "3px 8px"),
          textsize = "13px",
          direction = "auto"
        ),
        highlightOptions = highlightOptions(
          weight = 3,
          color = "#666",
          fillOpacity = 0.9,
          bringToFront = TRUE
        ),
        popup = ~paste("<strong>State:</strong>", SHP_NAME, "<br>",
                       "<strong>Total Offences in", input$year, ":</strong>", 
                       ifelse(is.na(Total_Offences), "N/A", format(Total_Offences, big.mark=",")))
      ) %>%
      addLegend(
        pal = pal,
        values = ~Total_Offences,
        opacity = 0.7,
        title = "Total Offences",
        position = "bottomright"
      )
  })
  
  # Reactive dataset for line chart based on selected state and offender type
  state_data <- reactive({
    req(nrow(data) > 0)
    if (is.null(input$map_shape_click)) {
      data %>% 
        filter(Youth.Offender.Type == input$offender_type) %>%
        group_by(Year) %>%
        summarise(Total_Offences = sum(Offences, na.rm = TRUE), .groups = "drop")
    } else {
      clicked_state <- input$map_shape_click$id
      data %>% 
        filter(State == clicked_state, Youth.Offender.Type == input$offender_type) %>%
        group_by(Year) %>%
        summarise(Total_Offences = sum(Offences, na.rm = TRUE), .groups = "drop")
    }
  })
  
  # Render the line chart (Plotly, no GIF)
  output$linePlot <- renderPlotly({
    df <- state_data()
    validate(need(nrow(df) > 0, "No time series data available."))
    
    ttl <- if (is.null(input$map_shape_click)) {
      "Total Youth Offender Offences Over Time (All States)"
    } else {
      paste("Offences Over Time in", input$map_shape_click$id)
    }
    
    plot_ly(df, x = ~Year, y = ~Total_Offences, type = "scatter", mode = "lines+markers") %>%
      layout(title = ttl, xaxis = list(dtick = 1))
  })
  
  # Render the heatmap for screen usage purposes (Plotly)
  output$heatmap_screen_usage <- renderPlotly({
    validate(need(nrow(data_screen_purpose) > 0, "Screen usage dataset is empty."))
    
    filtered_data_sp <- subset(data_screen_purpose, Year == input$screen_year)
    validate(need(nrow(filtered_data_sp) > 0, "No records for the selected year."))
    
    plot_ly(
      filtered_data_sp,
      x = ~State, y = ~Purpose, z = ~Percentage,
      type = "heatmap",
      text = ~paste0(Percentage, "%"),
      hoverinfo = "x+y+text"
    ) %>%
      layout(title = "Children's Screen Usage Purposes Across Australia")
  })
  
  # Render the scatter plot and Venn diagram for correlation analysis
  predicted_value <- reactiveVal(NULL)  # Store predicted value
  
  observeEvent(input$predict_button, {
    req(!is.null(lm_fit))
    Device_Users <- input$Device_Users_input
    predicted_youth_offenders <- predict(lm_fit, newdata = data.frame(Device_Users = Device_Users))
    predicted_value(data.frame(Device_Users = Device_Users, Youth_Offenders = as.numeric(predicted_youth_offenders)))
  })
  
  output$scatterPlot <- renderPlotly({
    validate(need(nrow(merged_data) > 0, "Merged dataset is empty. Check input files/columns."))
    
    # Base bubble scatter
    plt <- plot_ly(
      data = merged_data,
      x = ~Device_Users, y = ~Youth_Offenders,
      type = "scatter", mode = "markers",
      color = ~State,
      size = ~Device_Users, sizes = c(10, 40),
      marker = list(opacity = 0.7)
    ) %>%
      layout(
        title = "Bubble Plot: Screen Users vs Youth Offenders",
        xaxis = list(title = "Children Participating in Screen-based Activities"),
        yaxis = list(title = "Youth Offenders"),
        legend = list(orientation = "h")
      )
    
    # Regression line — prevent inheriting mappings (fix length mismatch)
    if (!is.null(lm_fit)) {
      xs <- seq(min(merged_data$Device_Users, na.rm = TRUE),
                max(merged_data$Device_Users, na.rm = TRUE), length.out = 100)
      ys <- predict(lm_fit, newdata = data.frame(Device_Users = xs))
      plt <- plt %>%
        add_lines(x = xs, y = ys, name = "Linear fit",
                  line = list(dash = "dash"), inherit = FALSE)
    }
    
    # Predicted point — also no inherit
    if (!is.null(predicted_value())) {
      p <- predicted_value()
      plt <- plt %>%
        add_markers(data = p,
                    x = ~Device_Users, y = ~Youth_Offenders,
                    name = "Predicted",
                    marker = list(symbol = "x", size = 12),
                    inherit = FALSE)
    }
    
    plt
  })
  
  output$vennAndTextPlot <- renderPlot({
    validate(need(nrow(merged_data) > 1, "Not enough data to compute correlation."))
    
    # Calculate correlation coefficient
    correlation_value <- suppressWarnings(cor(merged_data$Device_Users, merged_data$Youth_Offenders))
    shared_variance <- round((correlation_value^2) * 100, 2)
    
    # Create correlation metrics text plot (ggplot)
    correlation_text <- paste(
      "Correlation: ", round(correlation_value, 2), "\n",
      "Shared variance: ", shared_variance, "%\n",
      if (!is.null(lm_fit)) {
        cfs <- coef(lm_fit)
        paste0("y = ", round(cfs[1], 2), " + ", round(cfs[2], 4), " * x")
      } else {
        "Model unavailable"
      }, sep = ""
    )
    
    text_plot <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = correlation_text, hjust = 0.5, vjust = 0.5, size = 6, color = "#333333") +
      theme_void()
    
    # Venn diagram
    fit <- eulerr::euler(c("Screen Users" = 100,
                           "Youth Offenders" = 100,
                           "Screen Users&Youth Offenders" = shared_variance))
    venn_plot <- plot(fit,
                      fills = list(fill = c("#66c2a5", "#fc8d62"), alpha = 0.5),
                      labels = list(font = 2))
    
    # Arrange side-by-side safely inside renderPlot
    gridExtra::grid.arrange(venn_plot, text_plot, ncol = 2, widths = c(1, 1))
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)