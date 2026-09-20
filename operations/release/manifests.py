"""Render scoped installation objects; never apply them. Author: Zeno Ren."""
import argparse
import json


def objects(namespace,target_namespace,target,image,configmaps=(),secrets=(),lab=False):
    account='lb-release-controller'
    def role(name,ns,rules):return {'apiVersion':'rbac.authorization.k8s.io/v1','kind':'Role','metadata':{'name':name,'namespace':ns},'rules':rules}
    def binding(name,ns):return {'apiVersion':'rbac.authorization.k8s.io/v1','kind':'RoleBinding','metadata':{'name':name,'namespace':ns},
        'subjects':[{'kind':'ServiceAccount','name':account,'namespace':namespace}],
        'roleRef':{'apiGroup':'rbac.authorization.k8s.io','kind':'Role','name':name}}
    result=[{'apiVersion':'v1','kind':'Namespace','metadata':{'name':namespace}},
            {'apiVersion':'v1','kind':'ServiceAccount','metadata':{'name':account,'namespace':namespace}}]
    result.extend([role(account,namespace,[
        {'apiGroups':[''],'resources':['configmaps'],'verbs':['get','list','create','patch','update']},
        {'apiGroups':[''],'resources':['secrets'],'verbs':['get','create']},
        {'apiGroups':[''],'resources':['pods'],'verbs':['get','list']},
        {'apiGroups':[''],'resources':['events'],'verbs':['create']},
        {'apiGroups':['coordination.k8s.io'],'resources':['leases'],'verbs':['get','create','update']}]),binding(account,namespace)])
    rules=[{'apiGroups':['apps'],'resources':['deployments'],'resourceNames':[target],'verbs':['get','patch']},
           {'apiGroups':['apps'],'resources':['replicasets'],'verbs':['get','list']},
           {'apiGroups':[''],'resources':['pods'],'verbs':['get','list','create','patch','delete']},
           {'apiGroups':[''],'resources':['pods/exec'],'verbs':['create','get']},
           {'apiGroups':[''],'resources':['services'],'verbs':['get','list','patch']},
           {'apiGroups':['discovery.k8s.io'],'resources':['endpointslices'],'verbs':['get','list']},
           {'apiGroups':['networking.k8s.io'],'resources':['ingresses'],'verbs':['get','list']}]
    rules.append({'apiGroups':['autoscaling'],'resources':['horizontalpodautoscalers'],'verbs':['get','list']})
    for resource,names in [('configmaps',configmaps),('secrets',secrets)]:
        if names:rules.append({'apiGroups':[''],'resources':[resource],'resourceNames':list(names),'verbs':['get']})
    result.extend([role(account+'-target',target_namespace,rules),binding(account+'-target',target_namespace)])
    node_role=account+'-nodes-'+namespace
    result.extend([{'apiVersion':'rbac.authorization.k8s.io/v1','kind':'ClusterRole','metadata':{'name':node_role},
                    'rules':[{'apiGroups':[''],'resources':['nodes'],'verbs':['get']}]},
                   {'apiVersion':'rbac.authorization.k8s.io/v1','kind':'ClusterRoleBinding','metadata':{'name':node_role},
                    'subjects':[{'kind':'ServiceAccount','name':account,'namespace':namespace}],
                    'roleRef':{'apiGroup':'rbac.authorization.k8s.io','kind':'ClusterRole','name':node_role}}])
    labels={'app':account}
    args=['--namespace',namespace,'--targets',target_namespace+'/'+target]+(['--lab'] if lab else [])
    result.append({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':account,'namespace':namespace},
        'spec':{'replicas':2,'strategy':{'type':'RollingUpdate','rollingUpdate':{'maxSurge':0,'maxUnavailable':1}},
            'selector':{'matchLabels':labels},'template':{'metadata':{'labels':labels,
                'annotations':{'prometheus.io/scrape':'true','prometheus.io/port':'9000','prometheus.io/path':'/metrics'}},'spec':{
            'serviceAccountName':account,'terminationGracePeriodSeconds':20,
            'securityContext':{'runAsNonRoot':True,'runAsUser':1000,'seccompProfile':{'type':'RuntimeDefault'}},
            'affinity':{'podAntiAffinity':{'requiredDuringSchedulingIgnoredDuringExecution':[
                {'labelSelector':{'matchLabels':labels},'topologyKey':'kubernetes.io/hostname'}]}},
            'containers':[{'name':'controller','image':image,'args':args,
                'env':[{'name':'POD_UID','valueFrom':{'fieldRef':{'fieldPath':'metadata.uid'}}}],
                'ports':[{'containerPort':9000}],
                'resources':{'requests':{'cpu':'10m','memory':'96Mi'},'limits':{'cpu':'250m','memory':'256Mi'}},
                'securityContext':{'readOnlyRootFilesystem':True,'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']}},
                'readinessProbe':{'httpGet':{'path':'/health/ready','port':9000},'periodSeconds':3},
                'livenessProbe':{'httpGet':{'path':'/health/live','port':9000},'periodSeconds':10},
                'volumeMounts':[{'name':'tmp','mountPath':'/tmp'}]}],
            'volumes':[{'name':'tmp','emptyDir':{}}]}}}})
    result.append({'apiVersion':'policy/v1','kind':'PodDisruptionBudget','metadata':{'name':account,'namespace':namespace},
                   'spec':{'minAvailable':1,'selector':{'matchLabels':labels}}})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('namespace','target-namespace','target','image'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--configmap',action='append',default=[]);parser.add_argument('--secret',action='append',default=[])
    parser.add_argument('--lab',action='store_true');args=parser.parse_args()
    print(json.dumps({'apiVersion':'v1','kind':'List','items':objects(args.namespace,args.target_namespace,args.target,args.image,args.configmap,args.secret,args.lab)},indent=2))


if __name__=='__main__':main()
