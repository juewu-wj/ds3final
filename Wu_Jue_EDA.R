# Load packages -----------------------------------------------------------
library(tidyverse)
library(janitor)
library(skimr)
library(rsample)
library(corrplot)

# Data cleaning and processing ---------------------------------------------------------------
# Load data
load("data/unprocessed/hsls_16_student_v1_0.rdata")

# Select variables needed
hsls <- hsls_16_student_v1_0 %>% 
  select(X1RACE,
         X2PAREDU,
         X3TGPAENG,
         X3TGPAMAT,
         X3TGPASCI,
         X3TGPASOCST,
         X3TGPAENGIN,
         X3TCREDSTEM,
         X3TGPASTEM,
         X3TGPATOT,
         X3TGPAMTHAP,
         X3TGPASCIAP,
         X4ENTMJST,
         C2INFSTEM,
         P1HIMAJ1_STEM,
         P1HIMAJ2_STEM)

# Save small file
saveRDS(hsls, file = "data/unprocessed/hsls.rds")

# Read small file
hsls <- read_rds("data/unprocessed/hsls.rds")

# Rename variables
hsls <- hsls %>% 
  rename(race = X1RACE,
         paredu = X2PAREDU,
         gpa_eng = X3TGPAENG,
         gpa_mat = X3TGPAMAT,
         gpa_sci = X3TGPASCI,
         gpa_sosci = X3TGPASOCST,
         gpa_engin = X3TGPAENGIN,
         cred_stem = X3TCREDSTEM,
         gpa_stem = X3TGPASTEM,
         gpa_total = X3TGPATOT,
         gpa_mat_ap = X3TGPAMTHAP,
         gpa_sci_ap = X3TGPASCIAP,
         stem = X4ENTMJST,
         info_stem = C2INFSTEM,
         p1stem = P1HIMAJ1_STEM,
         p2stem = P1HIMAJ2_STEM)

skim_without_charts(hsls)

# Remove missing values for response variable: 
# only keep No, Yes, Undeclared
hsls <- hsls %>% 
  filter(stem == 0 | stem == 1 | stem == -1)

# Remove missing values for students without GPA 
# note: keeping missing values for engineering and AP because that is an indicator of STEM
hsls <- hsls %>% 
  filter(gpa_mat >= 0,
         gpa_eng >= 0,
         gpa_sci >= 0,
         gpa_sosci >= 0,
         gpa_stem >= 0,
         gpa_total >= 0)

# Recode some variables as factors
hsls$race <- as_factor(hsls$race)
hsls$paredu <- as_factor(hsls$paredu)
hsls$stem <- as_factor(hsls$stem)
hsls$info_stem <- as_factor(hsls$info_stem)
hsls$p1stem <- as_factor(hsls$p1stem)
hsls$p2stem <- as_factor(hsls$p2stem)

skim_without_charts(hsls)

# Collapse categories for p1stem, p2stem, info_stem
# recode as keep No, Yes, NA
hsls <- hsls %>% 
  mutate(p1stem = fct_collapse(p1stem,
                               "-9" = c("-9", "-8", "-7")),
         p2stem = fct_collapse(p2stem,
                               "-9" = c("-9", "-8", "-7")),
         info_stem = fct_collapse(info_stem,
                                  "-9" = c("-9", "-8", "-6")))
         
skim_without_charts(hsls)

# Recode gpa_engin, gpa_mat_ap, gpa_sci_ap as categorical because lots of missing
hsls <- hsls%>% 
  mutate(gpa_engin = ifelse(gpa_engin >= 0, "Yes", "No"),
         gpa_mat_ap = ifelse(gpa_mat_ap >= 0, "Yes", "No"),
         gpa_sci_ap = ifelse(gpa_sci_ap >= 0, "Yes", "No"))

hsls$gpa_engin <- as_factor(hsls$gpa_engin)
hsls$gpa_mat_ap <- as_factor(hsls$gpa_mat_ap)
hsls$gpa_sci_ap <- as_factor(hsls$gpa_sci_ap)

# Save cleaned data to a file
saveRDS(hsls, file = "data/processed/hsls.rds")


# EDA ---------------------------------------------------------------------
# Read in cleaned data
hsls <- read_rds("data/processed/hsls.rds")

# Overview
skim_without_charts(hsls)

# Set seed
set.seed(3)

# Split: 15% testing, 85% modeling (80% training, 20% EDA)
hsls_split_info <- hsls %>% 
  initial_split(prop = 0.85) 

hsls_split <- tibble(
  eda = hsls_split_info %>% training() %>% sample_frac(0.2) %>% list(),
  train = hsls_split_info %>% training() %>% setdiff(eda) %>% list(),
  test = hsls_split_info %>% testing() %>% list()
)

# Select EDA data
hsls_eda <- hsls_split %>% 
  pluck("eda", 1) 

# Overview of EDA data
skim_without_charts(hsls_eda)

# Corrplot
hsls_eda %>% 
  select(gpa_eng, gpa_mat, gpa_sci, gpa_sosci, gpa_stem, gpa_total) %>% 
  cor() %>% 
  corrplot()

# Univariate
ggplot(hsls_eda, aes(stem)) +
  geom_bar() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# Bivariate
# STEM and Math GPA
ggplot(hsls_eda, aes(stem, gpa_mat)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and English GPA
ggplot(hsls_eda, aes(stem, gpa_eng)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and Science GPA
ggplot(hsls_eda, aes(stem, gpa_sci)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and Social Science GPA
ggplot(hsls_eda, aes(stem, gpa_sosci)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and STEM GPA
ggplot(hsls_eda, aes(stem, gpa_stem)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and total GPA
ggplot(hsls_eda, aes(stem, gpa_total)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and STEM Course Credits
ggplot(hsls_eda, aes(stem, cred_stem)) +
  geom_boxplot() +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM"))

# STEM and Engineering GPA
ggplot(hsls_eda, aes(stem, fill = gpa_engin)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") 

ggplot(hsls_eda, aes(gpa_engin, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM"))

# STEM and AP Math
ggplot(hsls_eda, aes(stem, fill = gpa_mat_ap)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") 

ggplot(hsls_eda, aes(gpa_mat_ap, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM")) 

# STEM and AP Science
ggplot(hsls_eda, aes(stem, fill = gpa_sci_ap)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") 

ggplot(hsls_eda, aes(gpa_sci_ap, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM")) 

# STEM and Information on STEM
ggplot(hsls_eda, aes(stem, fill = info_stem)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") 

ggplot(hsls_eda, aes(info_stem, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM")) 

# STEM and Parent 1 STEM
ggplot(hsls_eda, aes(stem, fill = p1stem)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") +
  scale_fill_discrete(name = "Parent 1 Major",
                      labels = c("NA",
                                 "Non-STEM",
                                 "STEM"))

ggplot(hsls_eda, aes(p1stem, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM")) 

# STEM and Parent 2 STEM
ggplot(hsls_eda, aes(stem, fill = p2stem)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Intended College Major",
                   labels = c("Undecided", 
                              "Non-STEM",
                              "STEM")) +
  ylab("Percentage") +
  scale_fill_discrete(name = "Parent 2 Major",
                      labels = c("NA",
                                 "Non-STEM",
                                 "STEM"))

ggplot(hsls_eda, aes(p2stem, fill = stem)) +
  geom_bar(position = "fill") +
  ylab("Percentage") +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM")) 

# STEM and Race/Ethnicity
ggplot(hsls_eda, aes(race, fill = stem)) +
  geom_bar(position = "fill") +
  scale_x_discrete(labels = c("NA",
                              "Native",
                              "Asian",
                              "Black/African",
                              "Hispanic, no race specified",
                              "Hispanic, race specified",
                              "More than one race",
                              "Pacific Islander",
                              "White")) +
  ylab("Percentage") +
  coord_flip() +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM"))

# STEM and Parent Education
ggplot(hsls_eda, aes(paredu, fill = stem)) +
  geom_bar(position = "fill") +
  scale_x_discrete(name = "Parents' Highest Level of Education",
                   labels = c("NA",
                              "Less than high school",
                              "High School",
                              "Occupational school",
                              "Associate",
                              "Bachelor",
                              "Master",
                              "Doctoral")) +
  ylab("Percentage") +
  coord_flip() +
  scale_fill_discrete(name = "Intended College Major",
                      labels = c("Undecided", 
                                 "Non-STEM",
                                 "STEM"))
