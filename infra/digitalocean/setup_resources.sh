# DigitalOcean Setup Script (Conceptual)
# This script outlines the commands to provision the infrastructure defined in Blueprint §17.

# 1. Create VPC
doctl vpcs create --name emailops-vpc --region nyc3

# 2. Create Spaces Bucket (S3)
# Note: Spaces are region-specific but global namespace.
doctl registry create emailops-registry
# (Spaces creation usually done via UI or s3cmd/terraform, doctl support is limited for spaces creation)

# 3. Create Managed PostgreSQL with pgvector
doctl databases create emailops-db \
  --engine pg \
  --version 15 \
  --size db-s-1vcpu-2gb \
  --region nyc3 \
  --num-nodes 1

# Wait for DB to be ready, then enable pgvector
# PGPASSWORD=... psql ... -c "CREATE EXTENSION vector;"

# 4. Create Managed Valkey (Redis)
doctl databases create emailops-redis \
  --engine redis \
  --version 7 \
  --size db-s-1vcpu-1gb \
  --region nyc3 \
  --num-nodes 1

# 5. Create DOKS Cluster
doctl kubernetes cluster create emailops-k8s \
  --region nyc3 \
  --node-pool "name=worker-pool;size=s-2vcpu-4gb;count=3" \
  --vpc-uuid <VPC_UUID>

# 6. Configure kubectl
doctl kubernetes cluster kubeconfig save emailops-k8s
