import os
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession

cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")

processing_instance_type_param = ParameterString(name="ProcessingInstanceType")
processing_instance_count_param = ParameterInteger(name="ProcessingInstanceCount")
training_instance_type_param = ParameterString(name="TrainingInstanceType")
training_instance_count_param = ParameterInteger(name="TrainingInstanceCount")
model_approval_status_param = ParameterString(name="ModelApprovalStatus")
test_score_threshold_param = ParameterFloat(name="TestScoreThreshold")

s3_path = "s3://mlops-implementation/usecase-1/"
s3_url_param = ParameterString(name="S3URL", default_value=s3_path)
input_s3_url = os.path.join(s3_path, "input")
train_s3_url = os.path.join(s3_path, "train")
test_s3_url = os.path.join(s3_path, "test")

file_name = "heart.csv"
processing_container_base_path = "/opt/ml/processing"
processing_container_input_data_path = os.path.join(processing_container_base_path, "input")
processing_container_train_data_path = os.path.join(processing_container_base_path, "train")
processing_container_test_data_path = os.path.join(processing_container_base_path, "test")

print("processing_container_input_data_path: ", processing_container_input_data_path)
print("processing_container_train_data_path: ", processing_container_train_data_path)
print("processing_container_test_data_path: ", processing_container_test_data_path)

bucket = "mlops-implementation"
role = sagemaker.get_execution_role()
sagemaker_session = PipelineSession()

framework_version = "0.23-1"
processing_base_job_name = "mlops-1-processing-job"

sklearn_processing = SKLearnProcessor(
    framework_version=framework_version,
    instance_count=processing_instance_count_param,
    instance_type=processing_instance_type_param,
    volume_size_in_gb=10,
    base_job_name=processing_base_job_name,
    role=role,
    sagemaker_session=sagemaker_session
)

processing_step = ProcessingStep(
    name="sklearn-processing-step",
    processor=sklearn_processing,
    display_name="mlops-1-processing-step",
    description="A processing step in the MLOps pipeline for our implementation no. 1",
    inputs=[
        ProcessingInput(source=input_s3_url, destination=processing_container_input_data_path)
    ],
    outputs=[
        ProcessingOutput(
            output_name="train_data",
            source=processing_container_train_data_path,
            destination=train_s3_url
        ),
        ProcessingOutput(
            output_name="test_data",
            source=processing_container_test_data_path,
            destination=test_s3_url
        )
    ],
    code="preprocessing.py",
    cache_config=cache_config
)

framework_version = "0.23-1"
training_base_job_name = "mlops-1-training-job"

sklearn_estimator = SKLearn(
    base_job_name=training_base_job_name,
    entry_point="train.py",
    framework_version=framework_version,
    py_version='py3',
    role=role,
    sagemaker_session=sagemaker_session,
    instance_type=training_instance_type_param,
    output_path="s3://mlops-implementation/usecase-1/model/"
    # instance_count=training_instance_count_param
)

training_step = TrainingStep(
    name="sklearn-training-step",
    display_name="mlops-1-training-step",
    description="A training step in the MLOps pipeline for our implementation no. 1",
    estimator=sklearn_estimator,
    inputs={
        "train": 
        TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
    cache_config=cache_config
)

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.model_step import ModelStep


framework_version = "0.23-1"

image_uri_inference = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region="us-east-1",
    version=framework_version,
    py_version='py3',
    instance_type="ml.m5.xlarge",
    image_scope="inference"
)

model = SKLearnModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=role,
    framework_version=framework_version,
    py_version='py3',
    entry_point="train.py",
    image_uri=image_uri_inference
)

register_args = model.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    approval_status="Approved"
)

step_register = ModelStep(
    name=f"mlops-1-register",
    step_args=register_args
)

from sagemaker.workflow.pipeline import Pipeline

pipeline_name = "mlops-1-training-pipeline"

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_type_param,
        processing_instance_count_param,
        training_instance_type_param,
        training_instance_count_param,
        model_approval_status_param,
        test_score_threshold_param,
        s3_url_param
    ],
    steps=[
        processing_step,
        training_step,
        step_register
    ],
    sagemaker_session=sagemaker_session
)

pipeline.upsert(role_arn=role)

pipeline_step_names = [x.name for x in pipeline.steps]
print("pipeline_step_names: ", pipeline_step_names)

execution = pipeline.start(
    parameters = dict(
        ProcessingInstanceType="ml.m5.xlarge",
        ProcessingInstanceCount=1,
        TrainingInstanceType="ml.m5.xlarge",
        TrainingInstanceCount=1,
        ModelApprovalStatus="PendingManualApproval",
        TestScoreThreshold=0.8,
        S3URL="s3://mlops-implementation/usecase-1/input/heart.csv"
    )
)



