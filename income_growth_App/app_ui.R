# Load libraries so they are available
list.of.packages <- c("shiny")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library("shiny")

# We'll begin by defining some of the UI elements as variables
# This helps keep the code organized and easier to debug

# Define a variable `intro_panel` for your first page. It should be a
# `tabPanel()` with a title "Introduction" to represent the first tab.
# This layout will contain the following elements:
intro_panel <- tabPanel(
  "Introduction",
  
  # A `titlePanel()` with the text "Income Inequality"
  titlePanel("Income Inequality"),
  
  # A paragraph `p()` describing with the text: "The below diagram was created
  # by the New York Times to illustrate the increasing level of inequality in
  # the US."
  p("The below diagram was created by the New York Times to illustrate the
    increasing level of inequality in the US."),
  
  # An image `img()` to display. This content should have a `src` attribute of
  # "inequality.png" (which will refer to the file in the `www/` folder).
  img(src = "inequality.png"),
  
  # A paragraph containing a hyperlink `a()` to the source of the article at
  # https://www.nytimes.com/interactive/2017/08/07/opinion/leonhardt-income-inequality.html
  p(a(href = "https://www.nytimes.com/interactive/2017/08/07/opinion/leonhardt-income-inequality.html", "(source)")),
  
  # A paragraph containing a 1-2 sentence description of what the diagram
  # shows. Your paragraph should include both `strong()` and emphasized `em()`
  # text.
  p(
    "The graph shows how the richest 1% of people have seen their incomes jump",
    strong("6%"),
    "since 1980",
    "while the bottom 60% have barely seen any increase",
    em("(their income has stagnated).")
  ),
  
  p('Click on the "Growth Chart" tab to see an interactive diagram.')
)

# Next, we'll define a few UI elements to render in our second panel

# Define a variable `sidebar_content` as a `sidebarPanel()` UI element
# containing the following information:

sidebar_content <- sidebarPanel(
  # A `sliderInput()` for the 'percentile' value, labeled "Income Percentile".
  # This slider should let the user pick a range between 0 and 100
  sliderInput(
    inputId = "percentile",
    label = "Income Percentile", min = 0, max = 100, value = c(0, 100)
  )
)

# Define a variable `main_content` as a `mainPanel()` UI element
# containing the following information:
main_content <- mainPanel(
  # A `plotOutput()` element showing the 'plot' output (defined in the server)
  plotOutput("plot"),
  
  # A paragraph with a hyperlink to the data source
  # http://gabriel-zucman.eu/usdina/
  p(
    "Source:",
    a(
      href = "http://gabriel-zucman.eu/usdina/",
      "http://gabriel-zucman.eu/usdina/"
    )
  )
)


# Define a variable `growth_panel` for your second page. It should be a
# `tabPanel()` with a title "Growth Chart" to represent the second tab.
# This layout will contain the following elements:
growth_panel <- tabPanel(
  "Growth Chart",
  
  # A `titlePanel()` with the text "Income growth 1980-2014"
  titlePanel("Income growth 1980-2014"),
  
  # A `sidebarLayout()` to create two columns.
  # The sidebar layout will contain elements:
  sidebarLayout(
    # Your `sidebar_content`
    sidebar_content,
    
    # Your `main_content`
    main_content
  )
)


# Finally, define a `ui` variable, assigning it a `navbarPage()` layout.
# You will use `shinyUI()` to render this variable (in `app.R`)
# Give the layout a title of "Income Inequality".
# The layout should include the following elements:
# - Your `intro_panel`
# - Your `growth_panel`

ui <- navbarPage(
  "Income Inequality",
  intro_panel,
  growth_panel
)
