# server.R

# install packages if necessary
list.of.packages <- c("dplyr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(dplyr)

# Read in data
source('./scripts/build_map.R')
source('./scripts/build_scatter.R')
df <- read.csv('./data/electoral_college.csv', stringsAsFactors = FALSE)
state_codes <- read.csv('./data/state_codes.csv', stringsAsFactors = FALSE)

# Join together state.codes and df
joined_data <- left_join(df, state_codes, by="state")

# Compute the electoral votes per 100K people in each state
joined_data <- joined_data %>% mutate(ratio = votes/population * 100000)

# Start shinyServer
server <- function(input, output) { 
  
  # Render a plotly object that returns your map
  output$map <- renderPlotly({ 
    return(build_map(joined_data, input$mapvar))
  }) 
  
  # Render a plotly object that returns your scatter plot
  output$scatter <- renderPlotly({
    return(build_scatter(joined_data, input$search))
  })
}

