from setuptools import find_packages, setup

package_name = 'giim_seg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yake',
    maintainer_email='yake@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'giim_seg_node_service = giim_seg.giim_seg_node_service:main',
        	'srv_client = giim_seg.srv_client:main',
        	'pcl_best_target_Plane_Pose = giim_seg.pcl_best_target_Plane_Pose:main',
        	
            
            
        ],
    },
)
