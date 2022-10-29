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
<<<<<<< Updated upstream

### Bens Features Changed
- **Roof Style**: `string` --> `int`
  - 0: Flat
  - 1: Gable
  - 2: Gambrel
  - 3: Hip
  - 4: Mansard
  - 5: Shed

- **Roof Matl**: `string` --> `int`
  - 0: ClyTile 
  - 1: CompShg 
  - 2: Membran 
  - 3: Metal 
  - 4: Roll 
  - 5: Tar&Grv 
  - 6: WdShake 
  - 7: WdShngl

- **Exterior 1st**: `string` --> `int`
  - 0: AsbShng 
  - 1: AsphShn
  - 2: BrkComm
  - 3: BrkFace
  - 4: CBlock
  - 5: CmntBd
  - 6: HdBoard
  - 7: ImStucc
  - 8: MetalSd
  - 9: Other
  - 10: Plywood
  - 11: PreCast
  - 12: Stone
  - 13: Stucco
  - 14: VinylSd
  - 15: Wd Shng
  - 16: WdShing

- **Exterior 2nd**: `string` --> `int`
  - 0: AsbShng 
  - 1: AsphShn
  - 2: BrkComm
  - 3: BrkFace
  - 4: CBlock
  - 5: CmntBd
  - 6: HdBoard
  - 7: ImStucc
  - 8: MetalSd
  - 9: Other
  - 10: Plywood
  - 11: PreCast
  - 12: Stone
  - 13: Stucco
  - 14: VinylSd
  - 15: Wd Shng
  - 16: WdShing

- **Mas Vnr Type**: `string` --> `int`
  - 0: BrkCmn
  - 1: BrkFace
  - 2: CBlock
  - 3: None
  - 4: Stone

- **Exter Qual**: `string` --> `int`
  - 0: Ex
  - 1: Gd
  - 2: TA
  - 3: Fa
  - 4: Po

- **Exter Cond**: `string` --> `int`
  - 0: Ex
  - 1: Gd
  - 2: TA
  - 3: Fa
  - 4: Po

- **Foundation**: `string` --> `int`
  - 0: BrkTil
  - 1: CBlock
  - 2: PConc
  - 3: Slab
  - 4: Stone
  - 5: Wood

- **Bsmt Qual**: `string` --> `int`
  - 0: Ex
  - 1: Gd
  - 2: TA
  - 3: Fa
  - 4: Po
  - 5: NA

- **Bsmt Cond**: `string` --> `int`
  - 0: Ex
  - 1: Gd
  - 2: TA
  - 3: Fa
  - 4: Po
  - 5: NA

- **Bsmt Exposure**: `string` --> `int`
  - 0: Gd
  - 1: Av
  - 2: Mn
  - 3: No
  - 4: NA

- **BsmtFin Type 1**: `string` --> `int`
  - 0: GLQ
  - 1: ALQ
  - 2: BLQ
  - 3: Rec
  - 4: LwQ
  - 5: Unf
  - 6: NA

- **BsmtFin Type 2**: `string` --> `int`
  - 0: GLQ
  - 1: ALQ
  - 2: BLQ
  - 3: Rec
  - 4: LwQ
  - 5: Unf
  - 6: NA
  
- **Heating**: `string` --> `int`
  - 0: Floor
  - 1: GasA
  - 2: GasW
  - 3: Grav
  - 4: OthW
  - 5: Wall
=======
- CONTINUE FROM HERE
**Garage Qual**: Garage quality 
<!-- Not sure how this differs from Garage Cond so it could possibly be dropped? they are rated on the same scale?? -->
  - remapped from 0-5 for each value
    - 0: Ex	Excellent
    - 1: Gd	Good
    - 2: TA	Typical/Average
    - 3: Fa	Fair
    - 4: Po	Poor
    - 5: NA	No Garage
**Garage Cond**: Garage condition
  - remapped from 0-5 for each value
    - 0: Ex	Excellent
    - 1: Gd	Good
    - 2: TA	Typical/Average
    - 3: Fa	Fair
    - 4: Po	Poor
    - 5: NA	No Garage		
**Paved Drive**: Paved driveway
  - remapped from 0-2 for each value
    - 0: Y	Paved 
    - 1: P	Partial Pavement
    - 2: N	Dirt/Gravel
**Pool QC**: Pool quality
  - remapped from 0-4 for each value
    - 0: Ex	Excellent
    - 1: Gd	Good
    - 2: TA	Average/Typical
    - 3: Fa	Fair
    - 4: NA	No Pool
**Fence**: Fence quality
  - remapped from 0-4 for each value
    - 0: GdPrv	Good Privacy
    - 1: MnPrv	Minimum Privacy
    - 2: GdWo   Good Wood
    - 3: MnWw   Minimum Wood/Wire
    - 4: NA	    No Fence
**Misc Feature**: Miscellaneous feature not covered in other categories
  - remapped from 0-5 for each value
    - 0: Elev	Elevator
    - 1: Gar2	2nd Garage (if not described in garage section)
    - 2: Othr	Other
    - 3: Shed	Shed (over 100 SF)
    - 4: TenC	Tennis Court
    - 5: NA	  None
**Sale Type**: Type of sale
  - remapped from 0-9 for each value
    - 0: WD 	Warranty Deed - Conventional
    - 1: CWD	Warranty Deed - Cash
    - 2: VWD	Warranty Deed - VA Loan
    - 3: New	Home just constructed and sold
    - 4: COD	Court Officer Deed/Estate
    - 5: Con	Contract 15% Down payment regular terms
    - 6: ConLw	Contract Low Down payment and low interest
    - 7: ConLI	Contract Low Interest
    - 8: ConLD	Contract Low Down
    - 9: Oth	Other
**Sale Condition**:Condition of sale
  - remapped from 0-6 for each value
    - 0: Normal   Normal Sale
    - 1: Abnorml	Abnormal Sale -  trade, foreclosure, short sale
    - 3: AdjLand	Adjoining Land Purchase
    - 4: Alloca	  Allocation - two linked properties with separate deeds, typically condo with a garage unit	
    - 5: Family	  Sale between family members
    - 6: Partial	Home was not completed when last assessed (associated with New Homes)
>>>>>>> Stashed changes
