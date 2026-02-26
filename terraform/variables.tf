variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"  # Mumbai — closest to India
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"  # 2 vCPU, 2 GB RAM — modest CPU
}

variable "key_pair_name" {
  description = "Name of existing EC2 key pair for SSH access"
  type        = string
  default     = ""
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}

variable "repo_url" {
  description = "Git repository URL"
  type        = string
  default     = "https://github.com/example/crop-recommendation-system.git"
}

variable "model_version" {
  description = "Model version string"
  type        = string
  default     = "1.0.0"
}
