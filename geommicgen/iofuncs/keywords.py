"""
This module defines the Keyword class and its subclasses and the TopLevelReader class. These
are used to read the input file. A single TopLevelReader instance is created as a module
level variable containing all the allowed Keywords.

TO ADD A POSSIBLE KEYWORD, ADD IT TO THE INSTANCE OF THE TopLevelReader.
"""

import numpy as np

import microstructure.particle_classes as part_cls
import microstructure.phase as phase
import errors.error_classes as error_cls


class Keyword(object):
    """This is the class for keywords used in the input file.

    Attributes
    ----------
    name: str
        Name of the keyword.

    type: optional, {'float', 'int', 'bool', 'str', 'none'}
        Type of the value corresponding to the keyword. 'none' will set the value of the
        keyword as its name.

    Class Attributes
    ----------------
    input_reader: `.TopLevelReader`
        Object that keeps track of where in input file we are and looks for top level
        keywords.
    """

    def __init__(self, name, type=None, **kwargs):
        """
        Constructor for the Keyword class.

        Parameters
        ----------
        name: str
            Name of the keyword.

        keyword_group: {'PROBLEM_TYPE', 'N_DP_SAMPLES', 'MIC_GEN_PARAMETERS',
            'MIC_GEN_DESCRIPTORS', 'MESH_OPTIONS'}
            Group to wich the keyword belongs. Used for storage in the right variable.

        type: str, optional
            Type of the variable

        Keyword Arguments
        -----------------
        default_value: object
            Default value for the keyword
        """
        self.name = name
        self.type = type

    def readValue(self):
        """Read the value of the keyword."""
        line = Keyword.input_reader.input[Keyword.input_reader.i_line]
        try:
            if self.type == "float":
                value_str = line.split()[1:]
                if len(value_str) == 1:
                    final_val = float(value_str[0])
                else:
                    value_str = " ".join(value_str)
                    value_str = value_str.split(", ")
                    value_str[0] = value_str[0][1:]
                    value_str[-1] = value_str[-1][:-1]
                    # Remove squre brackets from vector
                    final_val = np.array([float(val) for val in value_str])
            elif self.type == "int":
                value_str = line.split()[1:]
                if len(value_str) == 1:
                    final_val = int(value_str[0])
                else:
                    value_str = " ".join(value_str)
                    value_str = value_str.split(", ")
                    value_str[0] = value_str[0][1:]
                    value_str[-1] = value_str[-1][:-1]
                    # Remove squre brackets from vector
                    final_val = np.array([int(val) for val in value_str])
            elif self.type == "bool":
                value_str = line.split()[1]
                if value_str == "True":
                    final_val = True
                elif value_str == "False":
                    final_val = False
                else:
                    raise ValueError
            elif self.type == "str":
                value_str = line.split()[1]
                final_val = value_str
            elif self.type == "none":
                final_val = self.name
            else:
                value_str = line.split()[1]
                final_val = value_str
        except ValueError:
            print(
                "In line {0} the value of the keyword has the wrong type.".format(line)
            )
            raise
        Keyword.input_reader.i_line += 1
        return final_val

    def isIn(self, line):
        """Check if the first string in the *line* is the keyword *self*."""

        isIn = line.split()[0].lower() == self.name.lower()

        return isIn


class KeywordTypeA(Keyword):
    """This the class for keywords formatted in the input file as::

        keyword.name val

    store as

    ``{'keyword.keyword_group':{keyword.name: val, other_keyword: val}``

    Attributes
    ----------
    keyword_group: str
        Used for storage.
    """

    def __init__(self, name, keyword_group, **kwargs):
        """
        Instanciate a `.KeywordTypeB` object.

        Parameters
        ----------
        name: str
            Name of the keyword.

        Keyword Arguments
        -----------------
        default_value: object
            Default value
        """
        super().__init__(name, **kwargs)
        self.keyword_group = keyword_group

        if "default_value" in kwargs:
            self.default_value = kwargs["default_value"]
            self.storeValue(self.default_value)

    def storeValue(self, val):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options.setdefault(self.keyword_group.lower(), {})
        Keyword.input_reader.all_options[self.keyword_group.lower()][
            self.name.lower()
        ] = val


class KeywordTypeB(Keyword):
    """This the class for keywords formatted in the input file as::

        keyword.name val

    store as

    ``{'keyword.name': val}``

    """

    def __init__(self, name, **kwargs):
        """
        Instanciate a `.KeywordTypeB` object.

        Parameters
        ----------
        name: str
            Name of the keyword.
        """
        super().__init__(name, **kwargs)

        if "default_value" in kwargs:
            self.default_value = kwargs["default_value"]
            self.storeValue(self.default_value)

    def storeValue(self, value):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options[self.name.lower()] = value


class KeywordTypeC(Keyword):
    """This the class for keywords formatted in the input file as::

        keyword.name
        header_key_1 val
        sub_key_1 val
        sub_key_2 val
        header_key_2 val
        sub_key_3 val
        sub_key_4 val

    store as


    ``{'keyword.name':
        {head_key_val_1: {sub_key_1: val, sub_key_2: val}}
        {head_key_val_2: {sub_key_3: val, sub_key_4: val}}}``

    Attributes
    ----------
    header_keys: set(`.Keyword`)
        Set containing the acceptable header keywords.

    sub_keys: set(`.Keyword`)
        Set containing the acceptable sub keywords.
    """

    def __init__(self, name, header_keys, sub_keys, **kwargs):
        """
        Instanciate a `.KeywordTypeC` object.

        Parameters
        ----------
        name: str
            Name of the keyword.

        header_keys: set(`.Keyword`)
            Set containing the acceptable header keywords.

        sub_keys: set(`.Keyword`)
            Set containing the acceptable sub keywords.
        """
        super().__init__(name, **kwargs)
        self.header_keys = header_keys
        self.sub_keys = sub_keys

    def readValue(self):
        """Read the values of the *self* keyword. """
        options = {}
        Keyword.input_reader.i_line += 1
        # Moving over the line containing top level keyword
        Keyword.input_reader.ignoreComments()
        # Ignore comments
        while Keyword.input_reader.i_line < len(Keyword.input_reader.input):
            line = Keyword.input_reader.input[Keyword.input_reader.i_line]
            # Current line
            if all(
                [
                    not keyword.isIn(line)
                    for keyword in self.header_keys.union(self.sub_keys)
                ]
            ):
                # If the current line doesn't contain a known keyword, exit the block
                break
            for header_keyword in self.header_keys:
                if header_keyword.isIn(line):
                    keyword_already_supplied = set()
                    current_header = header_keyword.readValue()
                    if current_header in options:
                        raise ValueError(
                            "The {0} group {1} was specified twice.".format(
                                self.name, current_header
                            )
                        )
                    options[current_header] = {}
                    break
            for sub_keyword in self.sub_keys:
                if sub_keyword.isIn(line):
                    if sub_keyword in keyword_already_supplied:
                        raise ValueError(
                            "The keyword {0} is supplied twice.".format(
                                sub_keyword.name
                            )
                        )
                    keyword_already_supplied.add(sub_keyword)
                    value = sub_keyword.readValue()
                    options[current_header][sub_keyword.name.lower()] = value
                    break
            Keyword.input_reader.ignoreComments()
            # Ignore comments

        return options

    def storeValue(self, val):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options[self.name.lower()] = val


class TopLevelReader:
    """This is the class for the reader that keeps of where we are in the input file and
        looks for top level keywords.

    Attributes
    ----------
    all_keywords: dict
        Dictionary whose keys are the keyword names and the corresponding values the
        keyword objects.

    i_line: int
        Current line of the input file.

    input: list(str)
        List of strings containing the input file.

    all_options: dict
        Dictionary where all the options are stored as they are read.

    top_level_keywords: set(`.Keyword`)
        Set containing the top level keywords.
    """

    def __init__(self):
        """Instanciate a `.TopLevelReader` object."""
        self.all_keywords = {}
        self.i_line = 0
        self.input = None
        self.all_options = {}
        self.top_level_keywords = set()
        Keyword.input_reader = self

    def ignoreComments(self):
        """Ignore comments, moving to the next line that doesn't contain a commment."""
        while True:
            if self.i_line >= len(self.input):
                break
            # Remaain inside the file
            line = self.input[self.i_line]
            # Save current line
            if (
                line.strip() == ""
                or line.startswith("#")
                or line.strip() == "[insert here]"
            ):
                # if the line is empty or a comment move on to the next
                self.i_line += 1
                # Move to the next line
                continue
            else:
                break

    def checkTopLevelKeywords(self):
        """Check if the current line contains a keyword, and read it is the case."""
        current_line_keyword = False
        # Flag for the presence of a keyword in the current line
        line = self.input[self.i_line]
        # Current line
        for possible_keyword in self.top_level_keywords:
            # Checking what is the current keyword
            if possible_keyword.isIn(line):
                current_line_keyword = True
                # General keyword has been found
                val = possible_keyword.readValue()
                # Read the value
                possible_keyword.storeValue(val)
                # Store the value
                break
        if not current_line_keyword:
            # No keyword was found in the current line
            raise ValueError(
                "Line {0} of input file does not contain a keyword: {1}".format(
                    self.i_line, line
                )
            )

    def moveAlong(self):
        """Move alogn the input file."""
        self.ignoreComments()
        while self.i_line < len(self.input):
            # Remaain inside the file
            self.checkTopLevelKeywords()
            self.ignoreComments()

    def readInputFile(self, input_file_path):
        """Read the input file at *input_file_path*."""
        with open(input_file_path, "r") as input:
            self.input = input.readlines()
            # Saving the contents of the input file
            self.i_line = 0
            # Initializing the line counter
            self.moveAlong()
            # Move along the file

    def addTopLevelKeyword(self, *args):
        """Add a top level keyword to the input reader."""
        for keyword in args:
            self.top_level_keywords.add(keyword)
            self.all_keywords[keyword.name] = keyword


def generateAllPossibleKeywordsFromParticleAttributes():
    """
    Generate all possible keywords from the attributes of the `.Particle` class and
    subclasses.
    """

    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    all_particle_sub_classes = get_all_subclasses(part_cls.Particle)
    all_phase_descriptor_sub_classes = get_all_subclasses(phase.PhaseDescriptor)
    keyword_set = set()
    for descriptor, (_, var_type) in part_cls.Particle.possible_parameters.items():
        # Volume fraction and number of particles
        keyword_set.add(Keyword(descriptor, type=var_type))
    for particle_type in all_particle_sub_classes:
        for descriptor, (_, var_type) in particle_type.possible_parameters.items():
            if descriptor in part_cls.Particle.possible_parameters:
                continue
            keyword_set.add(Keyword(descriptor, type=var_type))
            keyword_set.add(Keyword(descriptor + "_distribution", type="str"))
            for distribution in all_phase_descriptor_sub_classes:
                for parameter in distribution.parameters:
                    keyword_set.add(Keyword(descriptor + "_" + parameter, type="float"))

    return keyword_set


top_level_reader = TopLevelReader()
top_level_reader.addTopLevelKeyword(
    KeywordTypeA("Max_Residue_Per_Particle", "Mic_Gen_Parameters", type="float"),
    KeywordTypeA("Max_Step", "Mic_Gen_Parameters", type="int"),
    KeywordTypeA(
        "Max_Steps_To_Relax",
        "Mic_Gen_Parameters",
        default_value=0,
        type="int",
    ),
    KeywordTypeA(
        "Speed_Up_Scheme",
        "Mic_Gen_Parameters",
        default_value="Cell",
        type="str",
    ),
    KeywordTypeA(
        "Verlet_Factor",
        "Mic_Gen_Parameters",
        type="float",
        parent_keyword=("Speed_Up_Scheme", "Verlet"),
    ),
    KeywordTypeA(
        "dt",
        "Mic_Gen_Parameters",
        default_value=0.05,
        type="float",
    ),
    KeywordTypeA(
        "Save_History",
        "Mic_Gen_Parameters",
        default_value=False,
        type="bool",
    ),
    KeywordTypeA(
        "Type_Initial_Configuration",
        "Mic_Gen_Parameters",
        default_value="random",
        type="str",
    ),
    KeywordTypeA(
        "Thermostat",
        "Mic_Gen_Parameters",
        default_type="multi_temperature",
        type="str",
    ),
    KeywordTypeA(
        "Min_Distance",
        "Mic_Gen_Parameters",
        default_value=0,
        type="float",
    ),
    KeywordTypeA(
        "Initial_Temp",
        "Mic_Gen_Parameters",
        default_value=2.5e10,
        type="float",
    ),
    KeywordTypeA(
        "Min_Eq_Steps_At_Temp",
        "Mic_Gen_Parameters",
        default_value=25,
        type="int",
    ),
    KeywordTypeA(
        "Lowering_Temp_Criterion",
        "Mic_Gen_Parameters",
        default_value="original",
        type="str",
    ),
    KeywordTypeA(
        "Average_Window",
        "Mic_Gen_Parameters",
        default_value=25,
        type="int",
    ),
    KeywordTypeA(
        "Remesh",
        "Mic_Gen_Parameters",
        default_value=False,
        type="bool",
    ),
    KeywordTypeA(
        "Damping_Coeff",
        "Mic_Gen_Parameters",
        default_value=0,
        type="float",
    ),
    KeywordTypeA(
        "Particle_Mass_Opt",
        "Mic_Gen_Parameters",
        default_value="volume",
        type="str",
    ),
    KeywordTypeA(
        "Force_Option",
        "Mic_Gen_Parameters",
        default_value="intersection_area",
        type="str",
    ),
    KeywordTypeA(
        "Berendsen_Coeff",
        "Mic_Gen_Parameters",
        default_value=1e-2,
        type="float",
    ),
    KeywordTypeA(
        "Force_Rescale",
        "Mic_Gen_Parameters",
        default_value=False,
        type="bool",
    ),
    KeywordTypeA("Dir_Previous_Mic", "Mic_Gen_Parameters", type="str"),
    KeywordTypeA("RVE_Dimensions", "Mic_Gen_Parameters", type="float"),
)
# Generation parameters


top_level_reader.addTopLevelKeyword(
    KeywordTypeB("Problem_Type", type="int"),
    KeywordTypeB("N_DP_Samples", type="int"),
    KeywordTypeB(
        "Motion_Analysis",
        default_value=False,
        type="bool",
    ),
)
# General keywords

top_level_reader.addTopLevelKeyword(
    KeywordTypeC(
        "Mic_Gen_Descriptors",
        header_keys={Keyword("Phase")},
        sub_keys={
            Keyword("Phase_Type", type="int"),
            *generateAllPossibleKeywordsFromParticleAttributes(),
        },
    )
)
# Phase descriptors

top_level_reader.addTopLevelKeyword(
    KeywordTypeC(
        "Mesh_Options",
        header_keys={Keyword("femsh", type="none"), Keyword("rgmsh", type="none")},
        sub_keys={
            Keyword("Element_Type", type="str"),
            Keyword("Mesh_Size", type="float"),
            Keyword("N_Voxels_Dims", type="int"),
        },
    )
)
# Mesh generation parameters
