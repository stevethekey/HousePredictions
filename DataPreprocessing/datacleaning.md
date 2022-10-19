# Data Cleaning Documentation
## For Recursive Feature Elimination

### Introduction 
The purpose of this documentation is to give clarity as to
why certain features have been cleaned, and how we cleaned them.

When cleaning features, we did one of the following changes:
- Dropped the whole feature
- Changed the contents of a feature from `string` to `integer`

The reason for each change is documented per feature changed

### Features Changed
- **Order**: Dropped
  - **Order** is for the purpose of the table, not an actual attribute
- **PID**: Dropped
  - The **PID** of each house is arbitrary and unique
- **MS Zoning**: `string` --> `int`
  - remapped from 0-7 for each value
    - 0: A	Agriculture
    - 1: C	Commercial
    - 2: FV	Floating Village Residential
    - 3: I	Industrial
    - 4: RH	Residential High Density
    - 5: RL	Residential Low Density
    - 6: RP	Residential Low Density Park 
    - 7: RM	Residential Medium Density
- CONTINUE FROM HERE