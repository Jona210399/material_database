from typing import Literal

SEARCH_DICT = {
    "AUTHORS": None,  # BIBLIOGRAPHY : Authors name for the main (first) reference Text
    "ARTICLE": None,  #  BIBLIOGRAPHY : Title of article for the main (first) reference Text
    "PUBLICATIONYEAR": None,  #  BIBLIOGRAPHY : Year of publication of an article in the reference Numerical, integer
    "PAGEFIRST": None,  #  BIBLIOGRAPHY : First page number of an article in the referenceNumerical, integer
    "JOURNAL": None,  #  BIBLIOGRAPHY : Title of journal for the reference Text
    "VOLUME": None,  #  BIBLIOGRAPHY : Volume of the journal in the reference Numerical, integer
    "ABSTRACT": None,  #  BIBLIOGRAPHY : Abstract for the main (first) reference Text
    "KEYWORDS": None,  #  BIBLIOGRAPHY : Keywords for the main (first) reference Text
    "CELLVOLUME": None,  #  CELL SEARCH : Cell volumeNumerical, floating point
    "CALCDENSITY": None,  #  CELL SEARCH : Calculated density Numerical, floating poit
    "CELLPARAMETERS": None,  #  CELL SEARCH : Cell lenght a,b,c and angles alpha, beta, gamma separated by whitespace, i.e.: a b c alpha beta gamma, * if any value Numerical, floating point
    "SEARCH": None,  #  CELLDATACELL SEARCH : Restriction of cellparameters.experimental, reduced, standardized
    "STRUCTUREDFORMULA": None,  # A CHEMISTRY SEARCH : Search for typical chemical groups Text
    "CHEMICALNAME": None,  #  CHEMISTRY SEARCH : Search for (parts of) the chemical name Text
    "MINERALNAME": None,  #  CHEMISTRY SEARCH : Search for the mineral name Text
    "MINERALGROUP": None,  #  CHEMISTRY SEARCH : Search for the mineral group Text
    "ZVALUECHEMISTRY": None,  #  SEARCH :Number of formula units per unit cell Numerical, integer
    "ANXFORMULA": None,  #  CHEMISTRY SEARCH : Search for the ANX formula Text
    "ABFORMULA": None,  #  CHEMISTRY SEARCH : Search for the AB formula Text
    "FORMULAWEIGHT": None,  #  CHEMISTRY SEARCH : Search for the formula weight Numerical, floating point
    "NUMBEROFELEMENTS": None,  #  CHEMISTRY SEARCH : Search for number of elementsinteger
    "COMPOSITION": None,  #  CHEMISTRY SEARCH : Search for the chemical composition (including stochiometric coefficients and/or oxidation numbers: EL:Co.(min):Co.(max):Ox.(min):Ox.(max)with El=element, Co=coefficient, Ox=oxidation number) Text
    "COLLECTIONCODE": None,  #  DB INFO : ICSD collection codeNumerical, integer
    "PDFNUMBER": None,  #  DB INFO : PDF number as assigned by ICDD Text
    "RELEASE": None,  #  DB INFO : Release tagNumerical, integer, special format
    "RECORDINGDATE": None,  #  DB INFO : Recording date of an ICSD entry Numerical, integer, special format
    "MODIFICATIONDATE": None,  #  DB INFO : Modification date of an ICSD entry Numerical, integer, special format
    "COMMENT": None,  #  EXPERIMENTAL SEARCH : Search for a comment Text
    "RVALUE": None,  #  EXPERIMENTAL SEARCH : R-value of the refinement (0.00 ... 1.00) Numerical, floating point
    "TEMPERATURE": None,  #  EXPERIMENTAL SEARCH : Temperature of the measurement Numerical, floating point
    "PRESSURE": None,  #  EXPERIMENTAL SEARCH : Pressure during the measurement Numerical, floating point
    "SAMPLETYPE": None,  # EXPERIMENTAL SEARCH : Search for the sample type: powder, singlecrystal
    "RADIATIONTYPE": None,  # EXPERIMENTAL SEARCH : Search for the radiation type: xray, electrons, neutrons, synchotron
    "STRUCTURETYPE": None,  #  STRUCTURE TYPE : Search for predefined structure types directly Select one
    "SPACEGROUPSYMBOL": None,  #  SYMMETRY : Search for the space group symbol Text
    "SPACEGROUPNUMBER": None,  #  SYMMETRY : Search for the space group number Numerical, integer
    "BRAVAISLATTICE": None,  #  SYMMETRY : Select One: Primitive, a-centered, b-centered, c-centered, Body-centered, Rhombohedral, Face-centered Select one
    "CRYSTALSYSTEM": None,  #  SYMMETRY : Crystal system Select one
    "CRYSTALCLASS": None,  #  SYMMETRY : Search for the crystal class Text
    "LAUECLASS": None,  #  SYMMETRY : Search for predefined Laueclass: -1, -3, -3m, 2/m, 4/m, 4/mmm ,6/m 6/mmm ,m-3 ,m-3m ,mmm Select one
    "WYCKOFFSEQUENCE": None,  #  SYMMETRY : Search for the Wyckoff sequence Text
    "PEARSONSYMBOL": None,  #  SYMMETRY : Search for the Pearson symbol Text
    "INVERSIONCENTER": None,  #  SYMMETRY : Should inversion center be included? TRUE or FALSE
    "POLARAXIS": None,
}

SEARCH_DICT = {key.lower(): value for key, value in SEARCH_DICT.items()}


ContentType = Literal[
    "EXPERIMENTAL_INORGANIC",
    "EXPERIMENTAL_METALORGANIC",
    "THERORETICAL_STRUCTURES",
]

AvailableProperties = Literal[
    "CollectionCode",
    "HMS",
    "StructuredFormula",
    "StructureType",
    "Title",
    " Authors",
    "Reference",
    "CellParameter",
    "ReducedCellParameter",
    "StandardizedCellParameter",
    "CellVolume",
    "FormulaUnitsPerCell",
    "FormulaWeight",
    "Temperature",
    "Pressure",
    "RValue",
    "SumFormula",
    "ANXFormula",
    "ABFormula",
    "ChemicalName",
    "MineralName",
    "MineralGroup",
    "CalculatedDensity",
    "MeasuredDensity",
    "PearsonSymbol",
    "WyckoffSequence",
    "Journal",
    "Volume",
    "PublicationYear",
    "Page",
    "Quality",
]
