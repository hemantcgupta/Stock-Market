import React, { Component } from 'react';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import Constant from '../../Constants/validator';
import cookieStorage from '../../Constants/cookie-storage';
import api from '../../API/api'
import './UserManagement.scss';
import '../../ag-grid-table.scss';


class UserProfile extends Component {
    constructor(props) {
        super(props);
        this.state = {
            userData: [],
            systemmodule: ''

        };

    }

    componentWillMount() {
        const userData = JSON.parse(cookieStorage.getCookie('userDetails'))
        this.setState({ userData: userData, systemmodule: userData.systemmodule });
    }

    render() {
        const { userData, systemmodule } = this.state;
        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='user-module'>
                    <div className="clearfix"></div>
                    <div className="row">
                        <div className="col-md-12 col-sm-12 col-xs-12">
                            <div className="x_panel fade">
                                <div className="add">
                                    <h2>User Profile</h2>
                                    <div className="clearfix"></div>
                                </div>
                                <div>
                                    <div className="module-form " style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                                        <div className="form-group profile">
                                            <label htmlFor="name">Name :</label>
                                            <div className='details'>{userData.username}</div>
                                        </div>

                                        <div className="form-group profile">
                                            <label htmlFor="email">Email :</label>
                                            <div className='details'>{userData.email}</div>
                                        </div>

                                        <div className="form-group profile">
                                            <label htmlFor="role">Role :</label>
                                            <div className='details'>{userData.role}</div>
                                        </div>

                                        <div className="form-group profile">
                                            <label htmlFor="role">POS Access :</label>
                                            <div className='details'> {Constant.convertPOSAccess(userData.access)}</div>
                                        </div>

                                        <div className="form-group profile">
                                            <label htmlFor="role">Route Access :</label>
                                            <div className='details'> {Constant.convertRouteAccess(userData.route_access)}</div>
                                        </div>

                                        <div className="form-group profile" style={{ display: 'block' }}>
                                            <label htmlFor="role">System Modules :</label>
                                            <div className='sysModules-main'>
                                                {systemmodule ? systemmodule.map((modules) =>
                                                    <div className='sysModules'>
                                                        {modules}
                                                    </div>
                                                ) : ''}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
}

export default UserProfile;