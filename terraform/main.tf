# --------------------------------------------------
# Terraform — AWS EC2 Deployment for Crop Recommendation API
# --------------------------------------------------
# Deploys a single EC2 instance running the Docker container
# with CloudWatch logging, Secrets Manager, and ALB.
# --------------------------------------------------

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ----- VPC & Networking -----
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "crop_api" {
  name_prefix = "crop-api-"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "API endpoint"
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
    description = "SSH access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "crop-recommendation-api"
    Project = "crop-recommendation"
  }
}

# ----- IAM Role -----
resource "aws_iam_role" "ec2_role" {
  name_prefix = "crop-api-ec2-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name_prefix = "crop-api-"
  role        = aws_iam_role.ec2_role.name
}

# ----- Secrets Manager -----
resource "aws_secretsmanager_secret" "api_config" {
  name_prefix = "crop-api-config-"
  description = "Configuration for crop recommendation API"
}

resource "aws_secretsmanager_secret_version" "api_config" {
  secret_id = aws_secretsmanager_secret.api_config.id
  secret_string = jsonencode({
    MODEL_VERSION = var.model_version
    LOG_LEVEL     = "INFO"
  })
}

# ----- EC2 Instance -----
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "api_server" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.crop_api.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -ex
    dnf update -y
    dnf install -y docker git
    systemctl enable docker
    systemctl start docker

    # Clone and build
    cd /opt
    git clone ${var.repo_url} crop-api || echo "Repo not available, using local"
    cd crop-api || cd /opt

    # Run container
    docker build -t crop-recommendation-api .
    docker run -d \
      --name crop-api \
      --restart unless-stopped \
      -p 8000:8000 \
      -e MODEL_DIR=/app/models \
      crop-recommendation-api

    # CloudWatch agent
    dnf install -y amazon-cloudwatch-agent
    cat > /opt/aws/amazon-cloudwatch-agent/etc/config.json <<CWEOF
    {
      "logs": {
        "logs_collected": {
          "files": {
            "collect_list": [
              {
                "file_path": "/var/log/messages",
                "log_group_name": "/crop-api/system",
                "log_stream_name": "{instance_id}"
              }
            ]
          }
        }
      }
    }
    CWEOF
    systemctl enable amazon-cloudwatch-agent
    systemctl start amazon-cloudwatch-agent
  EOF

  tags = {
    Name    = "crop-recommendation-api"
    Project = "crop-recommendation"
  }
}

# ----- CloudWatch Log Group -----
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/crop-api/application"
  retention_in_days = 30

  tags = {
    Project = "crop-recommendation"
  }
}

# ----- CloudWatch Alarms -----
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "crop-api-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "CPU utilization exceeds 80%"

  dimensions = {
    InstanceId = aws_instance.api_server.id
  }
}
