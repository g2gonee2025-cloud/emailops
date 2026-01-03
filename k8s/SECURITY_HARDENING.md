# Kubernetes Security Hardening Guide

This document provides fixes for critical Kubernetes security vulnerabilities found in the emailops infrastructure.

## Overview

The following SonarQube rules have identified security gaps:

- **S6864**: Missing memory limits on containers
- **S6873**: Missing CPU requests
- **S6892**: Missing memory requests
- **S6865**: Service accounts not bound to RBAC roles
- **S6596**: Using "latest" image tags (unpredictable deployments)

## Priority Fixes

### 1. Add Resource Limits and Requests (S6864, S6873, S6892)

**Problem**: Containers without resource limits can cause:
- Memory exhaustion and OOM kills
- CPU starvation (noisy neighbor problem)
- Unpredictable pod eviction
- Poor cluster autoscaling behavior

**Fix**: Add `resources` section to all container specs:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-api
spec:
  template:
    spec:
      containers:
      - name: cortex-api
        image: cortex-api:v1.2.3  # Use specific version, not :latest
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
```

**Guidelines for Resource Values**:

| Component | Memory Request | Memory Limit | CPU Request | CPU Limit |
|-----------|----------------|--------------|-------------|----------|
| Redis cache | 64Mi | 128Mi | 100m | 200m |
| PostgreSQL (managed) | N/A (external) | N/A | N/A | N/A |
| cortex-api (FastAPI) | 256Mi | 512Mi | 500m | 1000m |
| cortex-worker (async) | 512Mi | 1Gi | 1000m | 2000m |
| kube-state-metrics | 64Mi | 128Mi | 100m | 200m |
| Streamlit UI | 128Mi | 256Mi | 250m | 500m |
| prometheus | 128Mi | 256Mi | 250m | 500m |
| grafana | 64Mi | 128Mi | 100m | 200m |

### 2. Service Account Security (S6865)

**Problem**: Pods with automounted service accounts not bound to RBAC roles can:
- Access Kubernetes API unexpectedly
- Escalate privileges
- Expose the cluster to security breaches

**Fix Option A: Disable Automounting (Recommended)**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cortex-worker
  namespace: default
automountServiceAccountToken: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-worker
spec:
  template:
    spec:
      serviceAccountName: cortex-worker
      # Explicitly disable if pod override needed
      automountServiceAccountToken: false
      containers:
      - name: worker
        image: cortex-worker:v1.2.3
```

**Fix Option B: Bind to Restrictive RBAC Role**

If Kubernetes API access is needed:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: cortex-worker-role
  namespace: default
rules:
# Only include necessary permissions
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cortex-worker-binding
  namespace: default
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: cortex-worker-role
subjects:
- kind: ServiceAccount
  name: cortex-worker
  namespace: default
```

### 3. Fixed Image Tags (S6596)

**Problem**: Using `:latest` tag causes:
- Non-reproducible deployments
- Unexpected breaking changes
- Difficulty debugging issues
- Failed rollbacks

**Fix: Use Specific Semantic Versions**

```yaml
# ❌ WRONG
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: gcr.io/project/cortex-api:latest  # Bad!

# ✅ CORRECT
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: gcr.io/project/cortex-api:v1.2.3  # Good!
```

**Version Pinning Strategy**:

1. Use semantic versioning: `v1.2.3`
2. Never use `:latest`, `:main`, `:master`
3. Use immutable image digests for production:

```yaml
image: gcr.io/project/cortex-api@sha256:abc123def456...  # Immutable!
```

4. Store image digests in ConfigMap for easy auditing:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: image-versions
  namespace: default
data:
  cortex-api: "gcr.io/project/cortex-api:v1.2.3@sha256:abc123..."
  cortex-worker: "gcr.io/project/cortex-worker:v1.2.3@sha256:def456..."
  redis: "redis:7.0.0@sha256:ghi789..."
```

## Complete Hardened Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cortex-api
  namespace: default
  labels:
    app: cortex-api
    version: v1.2.3
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: cortex-api
  template:
    metadata:
      labels:
        app: cortex-api
        version: v1.2.3
    spec:
      # Security: Use specific service account
      serviceAccountName: cortex-api
      automountServiceAccountToken: false  # S6865: Disable unless needed
      
      # Security: Run with non-root user
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      
      containers:
      - name: cortex-api
        # Security: Use specific image version (S6596)
        image: gcr.io/emailops/cortex-api:v1.2.3
        imagePullPolicy: IfNotPresent
        
        # Security: Add resource requests and limits (S6864, S6873, S6892)
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        
        # Security: Container-level security context
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # Ports
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        # Environment
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: OUTLOOKCORTEX_DB_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        
        # Volume mounts for temporary files
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: var-tmp
          mountPath: /var/tmp
      
      # Volumes
      volumes:
      - name: tmp
        emptyDir: {}
      - name: var-tmp
        emptyDir: {}
      
      # Pod disruption budget
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - cortex-api
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cortex-api
  namespace: default
automountServiceAccountToken: false
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: cortex-api-pdb
  namespace: default
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: cortex-api
```

## Validation Checklist

Before merging Kubernetes manifests, verify:

- [ ] All containers have `resources.requests` and `resources.limits` defined
- [ ] All containers use specific image versions (not `:latest`)
- [ ] All ServiceAccounts have `automountServiceAccountToken: false` unless explicitly needed
- [ ] If API access needed, ServiceAccount is bound to RBAC role with minimal permissions
- [ ] Pods run with `securityContext.runAsNonRoot: true`
- [ ] Pods have `livenessProbe` and `readinessProbe` defined
- [ ] No privileged containers or `CAP_SYS_ADMIN`
- [ ] Volume mounts are restricted and read-only where possible

## Implementation Commands

Apply security hardening to all manifests:

```bash
# Validate manifests before applying
kubectl apply -f k8s/ --dry-run=client -o yaml > /tmp/manifests-validated.yaml

# Review changes
kubectl diff -f /tmp/manifests-validated.yaml

# Apply changes
kubectl apply -f k8s/

# Verify resources were created
kubectl get pods -A -o custom-columns=NAME:.metadata.name,CPU_REQ:.spec.containers[*].resources.requests.cpu,MEM_REQ:.spec.containers[*].resources.requests.memory

# Audit for violations
kubectl get pods -A -o json | jq '.items[] | select(.spec.containers[].resources.limits == null)'
```

## Monitoring and Enforcement

### Install Kubernetes Policy Engine (Kyverno)

```bash
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno --namespace kyverno --create-namespace
```

### Enforce Resource Limits

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
spec:
  validationFailureAction: enforce
  rules:
  - name: check-limits
    match:
      resources:
        kinds:
        - Pod
    validate:
      message: "CPU and memory limits required"
      pattern:
        spec:
          containers:
          - resources:
              limits:
                memory: "?*"
                cpu: "?*"
              requests:
                memory: "?*"
                cpu: "?*"
```

## References

- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [NIST Application Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
