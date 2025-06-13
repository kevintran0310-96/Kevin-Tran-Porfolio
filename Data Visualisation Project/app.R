library(shiny)
library(leaflet)
library(dplyr)
library(sf)
library(ggplot2)
library(gganimate)
library(plotly)
library(gifski)
library(reshape2)
library(GGally)
library(gridExtra)
library(VennDiagram)
library(grid)
library(shinythemes)

# Load the cleaned data
data <- read.csv("Cleaned YO by Years.csv")

# Convert necessary columns to appropriate types
data$Year <- as.integer(data$Year)
data$Offences <- as.numeric(data$Offences)
data$Youth.Offender.Type <- as.factor(data$Youth.Offender.Type)

# Remove NA values from key columns to avoid issues during filtering
data <- na.omit(data)

# Load shapefile for state boundaries
shapefile_path <- "STE_2021_AUST_GDA2020.shp"
states_shape <- st_read(shapefile_path) %>% st_transform(crs = 4326) # Transform to WGS84

# Load the population data
population_data <- read.csv("YO Sex and Age by States.csv")

# Load the screen usage data
data_screen_purpose <- read.csv("Screen Purpose.csv", stringsAsFactors = FALSE)
# Convert Percentage to numeric and remove the percentage symbol
data_screen_purpose$Percentage <- as.numeric(sub("%", "", data_screen_purpose$Percentage))

# Load datasets for correlation plot
screen_data <- read.csv("Screen by State 2017.csv")
youth_offender_data <- read.csv("Youth Offender Rate in 2017.csv")

# Clean and prepare the data
if (nrow(screen_data) == 0 || nrow(youth_offender_data) == 0) {
  stop("One or both datasets are empty. Please check the data files.")
}

youth_offender_data$Youth.Offender.Number <- as.numeric(gsub(",", "", youth_offender_data$Youth.Offender.Number))

# Rename columns for clarity
colnames(youth_offender_data)[colnames(youth_offender_data) == "Year"] <- "State"
colnames(screen_data)[colnames(screen_data) == "Number.of.Children.Participating.in.Screen.based.Acitivities"] <- "Device_Users"

# Standardize 'State' names by trimming whitespaces
screen_data$State <- trimws(screen_data$State)
youth_offender_data$State <- trimws(youth_offender_data$State)

# Ensure numeric columns are properly converted
if (!is.numeric(screen_data$Device_Users)) {
  screen_data$Device_Users <- as.numeric(screen_data$Device_Users)
}

# Merge datasets by State
merged_data <- merge(screen_data, youth_offender_data, by = "State", all = TRUE)

# Remove rows with missing data
merged_data <- na.omit(merged_data)

# Check if merged data is empty after cleaning
if (nrow(merged_data) == 0) {
  stop("Merged dataset is empty after removing missing values. Please check the data files.")
}

# Rename columns for better readability
colnames(merged_data)[colnames(merged_data) == "Youth.Offender.Number"] <- "Youth_Offenders"

# Fit linear model to use for prediction
lm_fit <- lm(Youth_Offenders ~ Device_Users, data = merged_data)

# Define UI for the Shiny app
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Youth Offender Analysis Dashboard"),
  
  # Tabs layout
  tabsetPanel(
    # Tab 1: Project Introduction
    tabPanel(
      "Introduction",
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
  ),
    
    # Tab 2: Visualizations and Insights
    tabPanel(
      "Visualizations and Insights",
      fluidPage(
        fluidRow(
          column(12,
                 h2("Section 1: Youth Offender Offences Over Time"),
                 p("In the first section, we analyze the geographical distribution of youth offences as well as the temporal trends in these offences using a heatmap and an animated line chart. An additional aspect explored here is the breakdown of offence types, which helps to understand not just the quantity of youth crime but also its nature."),
                 p(HTML("<b>Key Insights:</b> The heatmap shows substantial variation across states, with <b>NSW and Victoria</b> showing consistently higher youth crime rates compared to other states like <b>Tasmania</b> or the <b>Northern Territory</b>. The animated line chart reveals fluctuations in offence rates over the years, with notable peaks, possibly linked to socio-economic events. <br>
                        Breaking down offence types, there is a <b>decline in serious crimes</b> such as homicide in NSW, but a <b>rise in minor and digital-related offences</b> nationwide, highlighting a shift towards online crimes among youth.")),
          )
        ),
        
        # Wrap the widgets in a consistent-sized column using fluidRow and column
        fluidRow(
          column(4,
                 wellPanel(
                   sliderInput("year", "Select Year:", 
                               min = min(data$Year, na.rm = TRUE), 
                               max = max(data$Year, na.rm = TRUE), 
                               value = max(data$Year, na.rm = TRUE), 
                               step = 1, sep = ""),
                   selectInput("offender_type", "Select Offender Type:", 
                               choices = unique(data$Youth.Offender.Type),
                               selected = unique(data$Youth.Offender.Type)[1])
                 )
          ),
          column(8,
                 fluidRow(
                   column(6, leafletOutput("map", height = "500px")),
                   column(6, imageOutput("linePlot", height = "600px"))
                 )
          )
        ),
        
        # Transition to Section 2
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
                 p(HTML("<b>Key Insights:</b> The <b>gender analysis</b> reveals a significant disparity, with <b>males consistently outnumbering females</b> in youth crime across all states and age groups. This is especially prominent in the <b>14-17 age group</b>, pointing to underlying socio-cultural factors influencing male youth behavior. States such as <b>Queensland</b> and <b>Western Australia</b> exhibit larger gender gaps, suggesting that regional differences may contribute to these trends, potentially driven by economic conditions or community environments.")),
          )
        ),
        
        fluidRow(
          column(4, align = "center",
                 wellPanel(
                   fluidRow(
                     column(6,
                            actionButton("male_button", label = "Male", icon = icon("male"), 
                                         style = "font-size: 20px; color: #FFFFFF; background-color: #007BFF; width: 90%; height: 70px")
                     ),
                     column(6,
                            actionButton("female_button", label = "Female", icon = icon("female"), 
                                         style = "font-size: 20px; color: #FFFFFF; background-color: #FF69B4; width: 90%; height: 70px")
                     )
                   ),
                   br(),
                   selectInput("state_input", "Select State:", 
                               choices = c("New South Wales", "Victoria", "Queensland", "South Australia", 
                                           "Western Australia", "Tasmania", "Northern Territory", "Australian Capital Territory")),
                   br(),
                   selectInput("year_input", "Select Year:", 
                               choices = unique(population_data$Year))
                 )
          ),
          column(8, plotlyOutput("population_plot", height = "600px"))
        ),
        
        # Transition to Section 3
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
                 p(HTML("<b>Key Insights:</b> The heatmap shows that activities such as '<b>calling parents/family</b>' and '<b>accessing the internet</b>' are consistently popular across states, reflecting the integral role of digital devices in communication and learning. States like <b>Western Australia (WA)</b> and <b>Queensland (QLD)</b> report higher levels of screen engagement compared to other states, suggesting local factors such as socio-economic conditions or cultural preferences influence these behaviors. Activities like '<b>listening to music</b>' or '<b>receiving calls</b>' are notably lower, indicating some variance in how different activities appeal to youths.")),
          )
        ),
        
        fluidRow(
          column(4,
                 wellPanel(
                   # Use a div to control the entire slider and play/pause button with some CSS styling
                   div(
                     style = "padding-bottom: 20px;",  # Add padding below to push the play/pause button down
                     sliderInput("screen_year", "Select Year:",
                                 min = min(data_screen_purpose$Year),
                                 max = max(data_screen_purpose$Year),
                                 value = min(data_screen_purpose$Year),  # Initial value
                                 step = 1, sep = "",
                                 animate = animationOptions(
                                   interval = 1000,
                                   playButton = tags$button(
                                     "Play", class = "btn btn-success", icon("play"), style = "margin-top: 10px;"
                                   ),
                                   pauseButton = tags$button(
                                     "Pause", class = "btn btn-warning", icon("pause"), style = "margin-top: 10px;"
                                   )
                                 ))
                   )
                 )
          ),
          column(8, plotlyOutput("heatmap_screen_usage"))
        ),
        
        # Transition to Section 4
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
                 p(HTML("<b>Key Insights:</b> The analysis reveals a strong positive correlation (<b>Pearson correlation coefficient of 0.94</b>) between the number of children engaged in screen-based activities and youth offender rates, suggesting that increased screen time is linked to a rise in youth crime. The <b>Venn diagram</b> indicates around <b>25-30%</b> of the variance in youth crime is shared with screen usage, highlighting the significance of digital engagement as a factor. However, the results also stress that other variables—such as socio-economic conditions—contribute significantly, meaning screen usage is one of many influences.")),
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
        
        fluidRow(
          column(12, plotOutput("vennAndTextPlot", height = "400px"))
        ),
        
        hr(),
        
        # Conclusion
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
    )
  )

# Define server logic for the Shiny app
server <- function(input, output, session) {
  
  # Reactive value to store selected sex
  selected_sex <- reactiveVal("Male")
  
  # Update selected sex when male/female button is clicked
  observeEvent(input$male_button, {
    selected_sex("Male")
    
    # Update button labels to indicate which one is selected
    updateActionButton(session, "male_button", label = "Male")
    updateActionButton(session, "female_button", label = "Female")
  })
  
  observeEvent(input$female_button, {
    selected_sex("Female")
    
    # Update button labels to indicate which one is selected
    updateActionButton(session, "female_button", label = "Female")
    updateActionButton(session, "male_button", label = "Male")
  })
  
  # Render the plot based on selected sex, state, and year
  output$population_plot <- renderPlotly({
    
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
    
    # Create a ggplot for Bubble Chart
    p <- ggplot(data_filtered, aes(x = Age, y = Population, size = Population, color = Age)) +
      geom_point(alpha = 0.7) +
      scale_size(range = c(5, 20)) +
      labs(title = paste(gender, "Youth Offender Number in", input$state_input, "for Year", input$year_input),
           x = "Age Group",
           y = "Youth Offender Number",
           size = "Youth Offender Number") +
      theme_minimal()
    
    # Convert ggplot to plotly for interactivity and customize the tooltip
    ggplotly(p, tooltip = c("x", "size"))
  })
  
  # Reactive filtered dataset for the latest year and offender type based on user input
  filtered_data <- reactive({
    data %>% 
      filter(Year == input$year, Youth.Offender.Type == input$offender_type) %>%
      group_by(State) %>%
      summarise(Total_Offences = sum(Offences))
  })
  
  # Render the leaflet heatmap
  output$map <- renderLeaflet({
    plot_data <- filtered_data()
    
    # Merge shapefile data with filtered data
    map_data <- states_shape %>% 
      left_join(plot_data, by = c("STE_NAME21" = "State"))
    
    # Create the leaflet heatmap with enhanced interactivity
    leaflet(map_data) %>% 
      addTiles() %>% 
      addPolygons(
        fillColor = ~colorNumeric("YlOrRd", Total_Offences, na.color = "#BDBDBD")(Total_Offences),
        fillOpacity = 0.7,
        color = "black",
        weight = 1,
        layerId = ~STE_NAME21, # Add layer ID for click interaction
        highlightOptions = highlightOptions(
          weight = 3,
          color = "#666",
          fillOpacity = 0.9,
          bringToFront = TRUE
        ),
        popup = ~paste("<strong>State:</strong>", STE_NAME21, "<br>",
                       "<strong>Total Offences in", input$year, ":</strong>", Total_Offences),
        labelOptions = labelOptions(
          style = list("font-weight" = "normal", padding = "3px 8px"),
          textsize = "13px",
          direction = "auto"
        )
      ) %>%
      addLegend(
        pal = colorNumeric("YlOrRd", plot_data$Total_Offences, na.color = "#BDBDBD"),
        values = ~Total_Offences,
        opacity = 0.7,
        title = "Total Offences",
        position = "bottomright"
      )
  })
  
  # Reactive dataset for line chart based on selected state and offender type
  state_data <- reactive({
    if (is.null(input$map_shape_click)) {
      # Default to show total number of youth offenders across all states
      data %>% 
        filter(Youth.Offender.Type == input$offender_type) %>%
        group_by(Year) %>%
        summarise(Total_Offences = sum(Offences))
    } else {
      # Filter for the clicked state
      clicked_state <- input$map_shape_click$id
      data %>% 
        filter(State == clicked_state, Youth.Offender.Type == input$offender_type) %>%
        group_by(Year) %>%
        summarise(Total_Offences = sum(Offences))
    }
  })
  
  # Render the line chart animation
  output$linePlot <- renderImage({
    plot_data <- state_data()
    
    # Ensure all years are included, even if there is no data for them
    all_years <- data.frame(Year = unique(data$Year))
    plot_data <- all_years %>%
      left_join(plot_data, by = "Year") %>%
      mutate(Total_Offences = ifelse(is.na(Total_Offences), 0, Total_Offences))
    
    # Create the animated line chart
    p <- ggplot(plot_data, aes(x = Year, y = Total_Offences)) +
      geom_line(color = "#005A9C", size = 1) +  # Create line connecting all points
      geom_point(color = "#FFA500", size = 2) +  # Adding points at each year for better visibility
      theme_minimal() +
      labs(title = if (is.null(input$map_shape_click)) {
        "Total Youth Offender Offences Over Time Across All States"
      } else {
        paste("Offences Over Time in", input$map_shape_click$id)
      },
      x = "Year",
      y = "Total Offences") +
      scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.1))) +
      scale_x_continuous(breaks = unique(data$Year)) +  # Ensure all years are displayed on the x-axis
      transition_reveal(Year)  # Use transition_reveal to make the line connect smoothly over time
    
    # Save animation as a gif to a temporary file
    outfile <- tempfile(fileext = ".gif")
    anim_save(outfile, animation = p, nframes = 100, renderer = gifski_renderer())
    
    # Return the path to the gif
    list(src = outfile, contentType = 'image/gif')
  }, deleteFile = TRUE)
  
  # Render the heatmap for screen usage purposes
  output$heatmap_screen_usage <- renderPlotly({
    # Filter data based on selected year
    filtered_data <- subset(data_screen_purpose, Year == input$screen_year)
    
    # Reshape data to wide format for heatmap
    heatmap_data <- dcast(filtered_data, Purpose ~ State, value.var = "Percentage")
    
    # Convert data to matrix for heatmap visualization
    heatmap_melt <- melt(heatmap_data, id.vars = "Purpose")
    
    # Create heatmap using ggplot
    p <- ggplot(filtered_data, aes(x = State, y = Purpose, fill = Percentage, frame = Year)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Percentage), color = "black", size = 3) +
      scale_fill_gradientn(colors = c("#ffffd9", "#41b6c4", "#081d58")) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Children's Screen Usage Purposes Across Australia",
           x = "State", y = "Purpose of Screen Usage", fill = "Percentage") +
      transition_states(Year, transition_length = 2, state_length = 1) +
      ease_aes('linear')  # Adds a smooth transition effect
    
    # Convert the ggplot object into a plotly interactive plot
    ggplotly(p)
  })
  
  # Render the scatter plot and Venn diagram for correlation analysis
  predicted_value <- reactiveVal(NULL)  # Store predicted value
  
  observeEvent(input$predict_button, {
    Device_Users <- input$Device_Users_input
    # Calculate predicted value using linear model
    predicted_youth_offenders <- predict(lm_fit, newdata = data.frame(Device_Users = Device_Users))
    predicted_value(data.frame(Device_Users = Device_Users, Youth_Offenders = predicted_youth_offenders))
  })
  
  output$scatterPlot <- renderPlotly({
    # Base scatter plot with trend line
    scatter_plot <- ggplot(merged_data, aes(x = Device_Users, y = Youth_Offenders, color = State)) +
      geom_point(aes(size = Device_Users), alpha = 0.6) +
      geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "#ff7f0e", linetype = "dashed", size = 1.5) +
      scale_size(range = c(3, 15)) +  # Increase the range of bubble sizes for better visual differentiation
      labs(
        title = "Bubble Plot: Screen Users vs Youth Offenders",
        x = "Children Participating in Screen-based Activities",
        y = "Youth Offenders",
        size = "Screen Users",
        color = "State"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  # Adjust title size
        axis.title = element_text(face = "bold"),
        legend.position = "bottom",  # Move legend to bottom
        legend.title = element_text(size = 10),  # Reduce legend title size
        legend.text = element_text(size = 8),  # Reduce legend text size
        plot.margin = margin(0, 0, 0, 0)  # Remove plot margin
      )
    
    # Add predicted point if available
    if (!is.null(predicted_value())) {
      prediction <- predicted_value()
      scatter_plot <- scatter_plot +
        geom_point(data = prediction, aes(x = Device_Users, y = Youth_Offenders), color = "red", size = 3) +  # Reduce point size to 3
        geom_text(data = prediction, aes(x = Device_Users, y = Youth_Offenders, label = paste0("Predicted Youth Offenders: ", round(Youth_Offenders))), 
                  vjust = -1, color = "red", size = 2.5, fontface = "bold")  # Reduce text size to 2.5
    }
    
    # Create a version of the plot without the size aesthetic for conversion to plotly
    scatter_plot_no_size <- scatter_plot + geom_point(alpha = 0.6) + geom_smooth(method = "lm", se = FALSE)
    
    # Convert scatter plot to plotly for interactivity, using hover tooltips
    plotly_plot <- ggplotly(scatter_plot_no_size, tooltip = c("x", "y", "color"))
    
    plotly_plot
  })
  
  output$vennAndTextPlot <- renderPlot({
    # Calculate correlation coefficient
    correlation_value <- cor(merged_data$Device_Users, merged_data$Youth_Offenders)
    shared_variance <- round((correlation_value^2) * 100, 2)
    
    # Create correlation metrics text plot
    correlation_text <- paste(
      "Correlation: ", round(correlation_value, 2), "\n",
      "Shared variance: ", shared_variance, "%\n",
      "y = ", round(coef(lm(Youth_Offenders ~ Device_Users, data = merged_data))[1], 2),
      " + ", round(coef(lm(Youth_Offenders ~ Device_Users, data = merged_data))[2], 2), " * x",
      sep = ""
    )
    
    text_plot <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = correlation_text, hjust = 0.5, vjust = 0.5, size = 6, color = "#333333") +
      theme_void() +
      theme(plot.margin = margin(0, 0, 0, 0)) # Remove margin for text plot
    
    # Create a static Venn diagram to represent shared variance
    venn <- draw.pairwise.venn(
      area1 = 100, # Total variance in Device_Users (as a percentage)
      area2 = 100, # Total variance in Youth_Offenders (as a percentage)
      cross.area = shared_variance, # Shared variance (as a percentage)
      category = c("Screen Users", "Youth Offenders"),
      fill = c("#66c2a5", "#fc8d62"),
      alpha = c(0.5, 0.5),
      cat.pos = c(-20, 20),
      cat.dist = 0.05,
      cex = 1.5,
      cat.cex = 1.5,
      fontface = "bold",
      cat.fontface = "bold",
      scaled = FALSE
    )
    
    # Render Venn diagram and text plot side by side
    grid.newpage()  # Start a new page for the grid arrangement
    grid.arrange(grobTree(venn), text_plot, ncol = 2, widths = c(1, 1))
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
