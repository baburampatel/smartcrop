output "api_public_ip" {
  description = "Public IP of the API server"
  value       = aws_instance.api_server.public_ip
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${aws_instance.api_server.public_ip}:8000"
}

output "api_docs" {
  description = "Swagger docs URL"
  value       = "http://${aws_instance.api_server.public_ip}:8000/docs"
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.api_server.id
}
