"""
Module reading the input data files of the keyword format, which the YAML one replaces.

This module defines the Keyword class and its subclasses and the TopLevelReader class,
which read a deck into the options the rest of the package was built from. It is kept
for the decks written before the YAML format, which `read_legacy_deck` reads and the
converter turns into input files; nothing else reads it.
"""

import numpy as np

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
import geommicgen.microstructure.particleclasses as part_cls
import geommicgen.microstructure.phase as phase


class Keyword:
    """This is the class for keywords used in the input file.

    Attributes
    ----------
    name: str
        Name of the keyword.

    type_str: optional, {'float', 'int', 'bool', 'str', 'int_list', 'float_list',
        'str_list', 'none'}
        Type of the value corresponding to the keyword. 'none' will set the value of the
        keyword as its name.

    Class Attributes
    ----------------
    input_reader: `.TopLevelReader`
        Object that keeps track of where in input file we are and looks for top level
        keywords.
    """

    def __init__(self, name, **kwargs):
        """
        Initialize an instance of the Keyword class.

        Parameters
        ----------
        name: str
            Name of the keyword.

        keyword_group: {'PROBLEM_TYPE', 'N_DP_SAMPLES', 'MIC_GEN_PARAMETERS',
            'MIC_GEN_DESCRIPTORS', 'MESH_OPTIONS'}
            Group to wich the keyword belongs. Used for storage in the right variable.

        type_str: str, optional
            Type of the variable

        Keyword Arguments
        -----------------
        default_value: object
            Default value for the keyword
        """
        self.name = name
        self.type_str = kwargs.get("type_str", None)

    def read_value(self):
        """Read the value of the keyword."""
        line = Keyword.input_reader.input[Keyword.input_reader.i_line]
        try:
            if self.type_str == "float":
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
            elif self.type_str == "int":
                value_str = line.split()[1:]
                if len(value_str) == 1:
                    final_val = int(value_str[0])
                else:
                    value_str = " ".join(value_str)
                    final_val = []
                    str_lists = value_str.split("]")[:-1]
                    for i_str_list in str_lists:
                        i_str_list = i_str_list.replace("[", " ")
                        i_str_list = i_str_list.replace("]", " ")
                        i_str_list = i_str_list.split(",")
                        final_val.append(
                            ([int(val) for val in i_str_list if val != " "])
                        )
            elif self.type_str == "bool":
                value_str = line.split()[1]
                if value_str == "True":
                    final_val = True
                elif value_str == "False":
                    final_val = False
                else:
                    raise ValueError
            elif self.type_str == "str":
                value_str = line.split()[1]
                final_val = value_str
            elif self.type_str == "int_list":
                value_str = " ".join(line.split()[1:])
                if "[" in value_str:
                    final_val = [
                        [
                            int(i_value)
                            for i_value in i_group.replace("[", " ").split(",")
                            if i_value.strip()
                        ]
                        for i_group in value_str.split("]")[:-1]
                    ]
                else:
                    final_val = [
                        int(i_value) for i_value in value_str.replace(",", " ").split()
                    ]
                # Written as [a, b], or as [a, b] [c, d] for several lists at once, in
                # which case each is read as a list of its own. A bare row of numbers
                # is read as one list
            elif self.type_str == "float_list":
                value_str = " ".join(line.split()[1:]).strip()
                if value_str.startswith("[") and value_str.endswith("]"):
                    value_str = value_str[1:-1]
                final_val = [
                    float(i_value) for i_value in value_str.replace(",", " ").split()
                ]
                # Written as [a, b] like every other list in the input file, or as a
                # single value, which is a list of one
            elif self.type_str == "str_list":
                value_str = " ".join(line.split()[1:]).strip()
                if value_str.startswith("[") and value_str.endswith("]"):
                    value_str = value_str[1:-1]
                final_val = [
                    i_value.strip()
                    for i_value in value_str.split(",")
                    if i_value.strip()
                ]
                # Written as [a, b] like every other list in the input file. A single
                # value and a comma separated list without the brackets are read too,
                # since the whole line is taken rather than the first word of it
            elif self.type_str == "none":
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

    def is_in(self, line):
        """Check if the first string in the *line* is the keyword *self*."""
        is_in = line.split()[0].lower() == self.name.lower()

        return is_in


class KeywordTypeA(Keyword):
    """This the class for keywords of the type A.

        Keyword of type A formatted in the input file as:

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
        Instanciate a `.KeywordTypeA` object.

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
            self.store_value(self.default_value)

    def store_value(self, val):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options.setdefault(self.keyword_group.lower(), {})
        Keyword.input_reader.all_options[self.keyword_group.lower()][
            self.name.lower()
        ] = val


class KeywordTypeB(Keyword):
    """This the class for keywords of the type B.

    Keyword of type B are formatted in the input file as:

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
            self.store_value(self.default_value)

    def store_value(self, value):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options[self.name.lower()] = value


class KeywordTypeC(Keyword):
    """This the class for keywords of the type C.

    Keyword of type C formatted in the input file as::

        keyword.name
        header_key_1 val
        sub_key_1 val
        sub_key_2 val
        header_key_2 val
        sub_key_3 val
        sub_key_4 val

    stored as::

        {'keyword.name':
            {head_key_val_1: {sub_key_1: val, sub_key_2: val}}
            {head_key_val_2: {sub_key_3: val, sub_key_4: val}}}

    Attributes
    ----------
    header_keys: set(`.Keyword`)
        Set containing the acceptable header keywords.

    sub_keys: dict
        Dictionary whose keys are the names of the header keywords and whose values are
        the sub keywords read under each of them. A sub keyword given under a header
        that does not take it is refused rather than ignored.

    all_sub_keys: set(`.Keyword`)
        Every sub keyword, whichever header it is read under.

    renamed_headers: dict
        The header each old name is read as, by the old name.
    """

    def __init__(self, name, header_keys, sub_keys, renamed_headers=None, **kwargs):
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

        renamed_headers: dict, optional
            The header each old name is read as, by the old name.
        """
        super().__init__(name, **kwargs)
        self.renamed_headers = renamed_headers or {}
        self.header_keys = header_keys | {
            Keyword(i_old, type_str="none") for i_old in self.renamed_headers
        }
        self.sub_keys = sub_keys
        self.all_sub_keys = set().union(*sub_keys.values())
        # Which header a line is under is known while it is being read, so the sub
        # keywords of that header are taken then; this is for recognising that a line
        # is still inside the block at all

    def read_value(self):
        """Read the values of the *self* keyword."""
        options = {}
        Keyword.input_reader.i_line += 1
        # Moving over the line containing top level keyword
        Keyword.input_reader.ignore_comments()
        # Ignore comments
        current_header = None
        while Keyword.input_reader.i_line < len(Keyword.input_reader.input):
            line = Keyword.input_reader.input[Keyword.input_reader.i_line]
            # Current line
            readable = self.header_keys
            if current_header is not None:
                readable = readable | self.all_sub_keys
            if not any(i_keyword.is_in(line) for i_keyword in readable):
                if any(
                    i_keyword.is_in(line)
                    for i_keyword in Keyword.input_reader.top_level_keywords
                ):
                    break
                # Another keyword ends the block
                raise ValueError(
                    "Line {0} of input file: {1} is not read under {2}, whose "
                    "headers are {3}.".format(
                        Keyword.input_reader.i_line + 1,
                        line.split()[0],
                        self.name,
                        ", ".join(sorted(self.sub_keys)),
                    )
                )
                # Any other line is a mistake in the block, rather than the end of it
            for header_keyword in self.header_keys:
                if header_keyword.is_in(line):
                    keyword_already_supplied = set()
                    name = self.renamed_headers.get(
                        header_keyword.name, header_keyword.name
                    )
                    current_sub_keys = self.sub_keys[name]
                    current_header = header_keyword.read_value()
                    if name != header_keyword.name:
                        current_header = name
                    # A header given under its old name is read as the one it names
                    if current_header in options:
                        raise ValueError(
                            "The {0} group {1} was specified twice.".format(
                                self.name, current_header
                            )
                        )
                    options[current_header] = {}
                    break
            for sub_keyword in self.all_sub_keys:
                if sub_keyword.is_in(line):
                    if sub_keyword not in current_sub_keys:
                        raise ValueError(
                            "The keyword {0} is not read under {1}.".format(
                                sub_keyword.name, current_header
                            )
                        )
                    # Refused rather than ignored: a keyword under the wrong header used
                    # to parse and then do nothing at all
                    if sub_keyword in keyword_already_supplied:
                        raise ValueError(
                            "The keyword {0} is supplied twice.".format(
                                sub_keyword.name
                            )
                        )
                    keyword_already_supplied.add(sub_keyword)
                    value = sub_keyword.read_value()
                    options[current_header][sub_keyword.name.lower()] = value
                    break
            Keyword.input_reader.ignore_comments()
            # Ignore comments

        return options

    def store_value(self, val):
        """Store the value of the keyword."""
        Keyword.input_reader.all_options[self.name.lower()] = val


class TopLevelReader:
    """Docstring for TopLevelReader class.

    This is the class for the reader that keeps of where we are in the input file and
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

    def ignore_comments(self):
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
            break

    def check_top_level_keywords(self):
        """Check if the current line contains a keyword, and read it is the case."""
        current_line_keyword = False
        # Flag for the presence of a keyword in the current line
        line = self.input[self.i_line]
        # Current line
        for possible_keyword in self.top_level_keywords:
            # Checking what is the current keyword
            if possible_keyword.is_in(line):
                current_line_keyword = True
                # General keyword has been found
                val = possible_keyword.read_value()
                # Read the value
                possible_keyword.store_value(val)
                # Store the value
                break
        if not current_line_keyword:
            # No keyword was found in the current line
            raise ValueError(
                "Line {0} of input file does not contain a keyword: {1}".format(
                    self.i_line + 1, line
                )
            )

    def move_along(self):
        """Move alogn the input file."""
        self.ignore_comments()
        while self.i_line < len(self.input):
            # Remaain inside the file
            self.check_top_level_keywords()
            self.ignore_comments()

    def read_input_file(self, input_file_path):
        """Read the input file at *input_file_path*."""
        with open(input_file_path, "r") as input_file:
            self.input = input_file.readlines()
            # Saving the contents of the input file
            self.i_line = 0
            # Initializing the line counter
            self.move_along()
            # Move along the file

    def add_top_level_keyword(self, *args):
        """Add a top level keyword to the input reader."""
        for keyword in args:
            self.top_level_keywords.add(keyword)
            self.all_keywords[keyword.name] = keyword


def generate_all_possible_keywords_from_particle_attributes():
    """
    Generate all possible keywords.

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
    for descriptor, (_, _, var_type) in part_cls.Particle.possible_parameters.items():
        # Volume fraction and number of particles
        keyword_set.add(Keyword(descriptor, type_str=var_type))
    for particle_type in all_particle_sub_classes:
        for descriptor, (_, _, var_type) in particle_type.possible_parameters.items():
            if descriptor in part_cls.Particle.possible_parameters:
                continue
            keyword_set.add(Keyword(descriptor, type_str=var_type))
            keyword_set.add(Keyword(descriptor + "_distribution", type_str="str"))
            for distribution in all_phase_descriptor_sub_classes:
                for parameter in distribution.parameters:
                    keyword_set.add(
                        Keyword(descriptor + "_" + parameter, type_str="float")
                    )

    return keyword_set


LEGACY_MESHERS = {"femsh": "gmsh", "rgmsh": "voxel"}
# The names the meshers were asked for by before they were asked for by their own, which
# a deck written then still gives

MESHER_KEYWORDS = {
    "gmsh": {
        "Mesh_Size": "float",
        "Elements_Per_Particle": "float",
        "Element_Type": "str",
    },
    "voxel": {"N_Voxels_Dims": "int_list"},
}
# The options of each mesher a deck gives, by keyword and the type it is read as

DISCRETISATION_KEYWORDS = {
    "Formats": "str_list",
    "File_Name": "str",
    "Periodic_Constraints": "bool",
    "Gauss_Points": "int",
    "Boundary_Type": "str",
}
# The options of a discretisation that belong to the formats it is written in and to its
# files, which a deck gives under either mesher

ANALYSIS_KEYWORDS = {
    "final_config": ("bool", False),
    "motion_analysis": ("bool", False),
    "voronoi_analysis": ("bool", False),
    "voronoi_type": ("str", "standard"),
    "n_surf_points": ("int", 10),
    "plot_voronoi": ("bool", False),
    "plot_imts": ("bool", False),
    "stat_nearest_neighbor": ("bool", False),
    "stat_ripleys_k": ("bool", False),
    "stat_two_pt_corr": ("bool", False),
    "stat_seed": ("int", None),
    "plot_rsa_vf_history": ("bool", False),
    "sim_gif": ("bool", False),
    "sim_gif_frame_duration": ("int", 200),
    "sim_gif_cleanup_frames": ("bool", True),
}
# The analyses a deck asks for, by keyword, with the type each is read as and the value
# it has when the deck does not give it

# The keywords of the meshes and the analyses as they were when the input file replaced
# the format, which is all a deck can give. They were built from what the meshers, the
# formats and the analyses declare; an option one of them declares later is given in an
# input file, and the reader of decks does not follow it


def build_reader():
    """
    Build a reader of decks, which has read none.

    The reader keeps the value of every keyword it reads, and gives the ones a deck
    leaves out the values an earlier deck gave them, so each deck is read by a reader
    of its own.

    Returns
    -------
    `.TopLevelReader`
        The reader, holding the default of every keyword that has one.
    """
    top_level_reader = TopLevelReader()

    # Generation parameters
    # ----------------------------------------------------------------------------------
    top_level_reader.add_top_level_keyword(
        KeywordTypeA(
            "Max_Residue_Per_Particle", "Mic_Gen_Parameters", type_str="float"
        ),
        KeywordTypeA("Max_Step", "Mic_Gen_Parameters", type_str="int"),
        KeywordTypeA(
            "Max_Steps_To_Relax",
            "Mic_Gen_Parameters",
            default_value=0,
            type_str="int",
        ),
        KeywordTypeA(
            "Speed_Up_Scheme",
            "Mic_Gen_Parameters",
            default_value="Cell",
            type_str="str",
        ),
        KeywordTypeA(
            "Verlet_Factor",
            "Mic_Gen_Parameters",
            type_str="float",
        ),
        KeywordTypeA(
            "dt",
            "Mic_Gen_Parameters",
            default_value=0.05,
            type_str="float",
        ),
        KeywordTypeA(
            "Save_History",
            "Mic_Gen_Parameters",
            default_value=False,
            type_str="bool",
        ),
        KeywordTypeA(
            "Type_Initial_Configuration",
            "Mic_Gen_Parameters",
            default_value="random",
            type_str="str",
        ),
        KeywordTypeA(
            "Thermostat",
            "Mic_Gen_Parameters",
            default_value="multi_temperature",
            type_str="str",
        ),
        KeywordTypeA(
            "Min_Distance",
            "Mic_Gen_Parameters",
            default_value=0,
            type_str="float",
        ),
        KeywordTypeA(
            "Initial_Temp",
            "Mic_Gen_Parameters",
            default_value=None,
            type_str="float",
        ),
        KeywordTypeA(
            "initial_vel_coeff",
            "Mic_Gen_Parameters",
            default_value=0.1,
            type_str="float",
        ),
        KeywordTypeA(
            "Min_Eq_Steps_At_Temp",
            "Mic_Gen_Parameters",
            default_value=25,
            type_str="int",
        ),
        KeywordTypeA(
            "final_overlap_check",
            "Mic_Gen_Parameters",
            default_value=False,
            type_str="bool",
        ),
        KeywordTypeA(
            "Lowering_Temp_Criterion",
            "Mic_Gen_Parameters",
            default_value="ratio_in_out",
            type_str="str",
        ),
        KeywordTypeA(
            "Average_Window",
            "Mic_Gen_Parameters",
            default_value=25,
            type_str="int",
        ),
        KeywordTypeA(
            "Damping_Coeff",
            "Mic_Gen_Parameters",
            default_value=0,
            type_str="float",
        ),
        KeywordTypeA(
            "Particle_Mass_Opt",
            "Mic_Gen_Parameters",
            default_value="radius",
            type_str="str",
        ),
        KeywordTypeA(
            "Force_Option",
            "Mic_Gen_Parameters",
            default_value="intersection_length",
            type_str="str",
        ),
        KeywordTypeA(
            "Berendsen_Coeff",
            "Mic_Gen_Parameters",
            default_value=1e-2,
            type_str="float",
        ),
        KeywordTypeA(
            "Force_Rescale",
            "Mic_Gen_Parameters",
            default_value=False,
            type_str="bool",
        ),
        KeywordTypeA(
            "Max_Ratio_Osc",
            "Mic_Gen_Parameters",
            default_value=2,
            type_str="int",
        ),
        KeywordTypeA(
            "dt_adapt",
            "Mic_Gen_Parameters",
            default_value=True,
            type_str="bool",
        ),
        KeywordTypeA(
            "temp_low_ratio",
            "Mic_Gen_Parameters",
            default_value=1 / 4,
            type_str="float",
        ),
        KeywordTypeA(
            "offset",
            "Mic_Gen_Parameters",
            default_value=True,
            type_str="bool",
        ),
        KeywordTypeA(
            "fixed_seed",
            "Mic_Gen_Parameters",
            default_value=None,
            type_str="int",
        ),
        KeywordTypeA(
            "Inner_Phase_Max_Step",
            "Mic_Gen_Parameters",
            default_value=200,
            type_str="int",
        ),
        KeywordTypeA("RVE_Dimensions", "Mic_Gen_Parameters", type_str="float"),
    )

    # Post Processing
    # ----------------------------------------------------------------------------------
    top_level_reader.add_top_level_keyword(
        *(
            KeywordTypeA(
                i_name, "post_proc", default_value=i_default, type_str=i_type
            )
            for i_name, (i_type, i_default) in ANALYSIS_KEYWORDS.items()
        )
    )

    # General keywords
    # ----------------------------------------------------------------------------------
    top_level_reader.add_top_level_keyword(
        KeywordTypeB("Problem_Type", type_str="int"),
        KeywordTypeB("N_DP_Samples", type_str="int"),
        KeywordTypeB("save_min", type_str="bool", default_value=False),
    )

    # Phase descriptors
    # ----------------------------------------------------------------------------------
    top_level_reader.add_top_level_keyword(
        KeywordTypeC(
            "Mic_Gen_Descriptors",
            header_keys={Keyword("Phase")},
            sub_keys={
                "Phase": {
                    Keyword("Phase_Type", type_str="int"),
                    Keyword("inner_phase", type_str="bool"),
                    Keyword("outer_phase", type_str="int"),
                    *generate_all_possible_keywords_from_particle_attributes(),
                }
            },
        )
    )

    # Mesh generation parameters
    # ----------------------------------------------------------------------------------
    discretisation = {
        Keyword(i_name, type_str=i_type)
        for i_name, i_type in DISCRETISATION_KEYWORDS.items()
    }
    sub_keys = {
        i_mesher: discretisation
        | {Keyword(j_name, type_str=j_type) for j_name, j_type in i_options.items()}
        for i_mesher, i_options in MESHER_KEYWORDS.items()
    }
    top_level_reader.add_top_level_keyword(
        KeywordTypeC(
            "Mesh_Options",
            header_keys={Keyword(i_name, type_str="none") for i_name in sub_keys},
            sub_keys=sub_keys,
            renamed_headers=LEGACY_MESHERS,
        )
    )
    # One header per mesher, under its own name, each reading the options of that
    # mesher. The finite element mesh used to be asked for as femsh and the grid as
    # rgmsh, which a deck written then is still read with

    return top_level_reader


def read_legacy_deck(path):
    """
    Read a deck.

    Parameters
    ----------
    path: str
        Path of the deck.

    Returns
    -------
    dict
        The options the deck gives, every keyword it leaves out at its default, with a
        mesher asked for by its old name under its own.
    """
    reader = build_reader()
    reader.read_input_file(path)
    options = reader.all_options
    options["mic_gen_parameters"]["sim_gif"] = options["post_proc"]["sim_gif"]
    # The frames of the GIF are drawn by the generation, which is told to as an input
    # file tells it

    return options
