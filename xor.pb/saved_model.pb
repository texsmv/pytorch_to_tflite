?r
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	"serve*2.4.12unknown8?X
f
ConstConst*
_output_shapes

:*
dtype0*)
value B"7,S????$Li????
X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"?CƿK?w?
`
Const_2Const*
_output_shapes

:*
dtype0*!
valueB"l9???a)?
T
Const_3Const*
_output_shapes
:*
dtype0*
valueB*`?>

NoOpNoOp
?
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*{
valuerBp Bj
X
handlers
outputs
initializer_dict
handler_variables

signatures


 
 
 
 
 
f
serving_default_inputPlaceholder*
_output_shapes

:*
dtype0*
shape
:
?
PartitionedCallPartitionedCallserving_default_inputConstConst_1Const_2Const_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference_signature_wrapper_54
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference__traced_save_110
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_restore_120?K
?
k
__inference__traced_save_110
file_prefix
savev2_const_4

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_4"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
?
u
 __inference_signature_wrapper_54	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity?
PartitionedCallPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? * 
fR
__inference___call___392
PartitionedCallc
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*=
_input_shapes,
*::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
?
p
__inference___call___83	
input
transpose_x
mul_1_y
transpose_1_x
mul_3_y
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Constu
flatten/ReshapeReshapeinputflatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permr
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*
_output_shapes

:2
	transposel
MatMulMatMulflatten/Reshape:output:0transpose:y:0*
T0*
_output_shapes

:2
MatMulS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/x\
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_1/xU
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:2
mul_1P
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:2
addq
onnx_tf_prefix_Sigmoid_1Sigmoidadd:z:0*
T0*
_output_shapes

:2
onnx_tf_prefix_Sigmoid_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeonnx_tf_prefix_Sigmoid_1:y:0flatten_1/Const:output:0*
T0*
_output_shapes

:2
flatten_1/Reshapeu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetranspose_1_xtranspose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1t
MatMul_1MatMulflatten_1/Reshape:output:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xd
mul_2Mulmul_2/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:2
mul_2W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_3/xU
mul_3Mulmul_3/x:output:0mul_3_y*
T0*
_output_shapes
:2
mul_3V
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:2
add_1T
IdentityIdentity	add_1:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*=
_input_shapes,
*::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
?
p
__inference___call___39	
input
transpose_x
mul_1_y
transpose_1_x
mul_3_y
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Constu
flatten/ReshapeReshapeinputflatten/Const:output:0*
T0*
_output_shapes

:2
flatten/Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permr
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*
_output_shapes

:2
	transposel
MatMulMatMulflatten/Reshape:output:0transpose:y:0*
T0*
_output_shapes

:2
MatMulS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
mul/x\
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_1/xU
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:2
mul_1P
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:2
addq
onnx_tf_prefix_Sigmoid_1Sigmoidadd:z:0*
T0*
_output_shapes

:2
onnx_tf_prefix_Sigmoid_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeonnx_tf_prefix_Sigmoid_1:y:0flatten_1/Const:output:0*
T0*
_output_shapes

:2
flatten_1/Reshapeu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permz
transpose_1	Transposetranspose_1_xtranspose_1/perm:output:0*
T0*
_output_shapes

:2
transpose_1t
MatMul_1MatMulflatten_1/Reshape:output:0transpose_1:y:0*
T0*
_output_shapes

:2

MatMul_1W
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xd
mul_2Mulmul_2/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:2
mul_2W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_3/xU
mul_3Mulmul_3/x:output:0mul_3_y*
T0*
_output_shapes
:2
mul_3V
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:2
add_1T
IdentityIdentity	add_1:z:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*=
_input_shapes,
*::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
?
E
__inference__traced_restore_120
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_defaultw
.
input%
serving_default_input:0)
output
PartitionedCall:0tensorflow/serving/predict:?	
?
handlers
outputs
initializer_dict
handler_variables

signatures
__call__
gen_tensor_dict"
_generic_user_object
$
"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
,
	serving_default"
signature_map
 "
trackable_dict_wrapper
?2?
__inference___call___83?
???
FullArgSpec
args?
jself
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec!
args?
jself
j
input_dict
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference_signature_wrapper_54input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3y
__inference___call___83^
.?+
? 
$?!

input?
input"&?#
!
output?
output?
 __inference_signature_wrapper_54^
.?+
? 
$?!

input?
input"&?#
!
output?
output