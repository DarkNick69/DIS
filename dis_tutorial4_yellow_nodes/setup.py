from setuptools import find_packages, setup

package_name = 'dis_tutorial4_yellow_nodes'

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
    maintainer='nickstab',
    maintainer_email='nikosstab@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'object_detection_node = dis_tutorial4_yellow_nodes.object_detection_node:main',
        'object_localizer_node = dis_tutorial4_yellow_nodes.object_localizer_node:main',
        'greeting_service = dis_tutorial4_yellow_nodes.greeting_service:main',
        'ring_color_announce_service = dis_tutorial4_yellow_nodes.ring_color_announce_service:main',
    ],
  },
)
