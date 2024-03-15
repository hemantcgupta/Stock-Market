import React, { Component } from 'react';
import { Redirect } from "react-router-dom";
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Access from "../../Constants/accessValidation";
import Loader from '../../Component/Loader';
import './UserManagement.scss';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import '../../ag-grid-table.scss';
import Spinners from "../../spinneranimation";
import Swal from 'sweetalert2';
import api from '../../API/api'


const apiServices = new APIServices();

const removeQuotes = (string) => {
    return string.substring(1, string.length - 1)
}

class SearchUser extends Component {
    constructor(props) {
        super(props);
        this.gridApi = null;
        this.state = {
            userName: '',
            email: '',
            role: '',
            userColumn: [
                { headerName: 'Id', field: 'id' },
                { headerName: 'User Name', field: 'user_name', tooltipField: 'user_name' },
                { headerName: 'Email', field: 'email', tooltipField: 'email' },
                { headerName: 'Role', field: 'role', tooltipField: 'role' },
                { headerName: 'POS Access', field: 'POS Access', tooltipField: 'POS Access' },
                { headerName: 'Route Access', field: 'Route Access', tooltipField: 'Route Access' },
                { headerName: 'Actions', field: '', cellRenderer: (params) => this.actions(params), width: 250 }
            ],
            userData: [],
            loading: false
        };

    }

    componentDidMount() {
        if (Access.accessValidation('Users', 'searchUser')) {
            this.getUserList();
        } else {
            this.props.history.push('/404')
        }
    }

    getUserList = () => {
        this.setState({ loading: true, userData: [] })
        const { userName, email, role } = this.state;
        const params = {};
        if (userName) {
            params['username'] = userName
        }
        if (email) {
            params['email'] = email
        }
        if (role) {
            params['role'] = role
        }
        const serialize = (params) => {
            var str = [];
            for (var p in params)
                if (params.hasOwnProperty(p)) {
                    str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
                }
            return str.join("&");
        }
        api.get(`userswithsearch?${serialize(params)}`, 'hide')
            .then((response) => {
                this.setState({ loading: false })
                if (response) {
                    if (response.data.response.length > 0) {
                        let userData = [];
                        let self = this;
                        response.data.response.forEach(function (key) {
                            userData.push({
                                "id": key.id,
                                "user_name": key.username,
                                "email": key.email,
                                "role": key.role,
                                "POS Access": self.convertPOSAccess(key.access),
                                "Route Access": self.convertRouteAccess(key.route_access)
                            });
                        });
                        this.setState({ userData: userData });
                    }
                }
            })
            .catch((err) => {
                console.log('rahul err', err)
                this.setState({ loading: false })
            })
    }

    convertPOSAccess(access) {
        let pos_access = access.split('#')
        let base_access = 'Network'

        if (pos_access[1] !== undefined) {
            if (pos_access[1] !== '*') {
                base_access = removeQuotes(pos_access[1])
            }
        }
        if (pos_access[2] !== undefined) {
            if (pos_access[2] !== '*') {
                base_access = `${base_access} > ${removeQuotes(pos_access[2])}`
            }
        }
        if (pos_access[3] !== undefined) {
            if (pos_access[3] !== '*') {
                base_access = `${base_access} > ${removeQuotes(pos_access[3])}`
            }
        }
        return base_access;
    }

    convertRouteAccess(access) {

        let route_access = access !== null ? access : {}
        let base_access = 'Network';

        if (Object.keys(route_access).length > 0) {
            if ((route_access).hasOwnProperty('selectedRouteGroup')) {
                base_access = `${route_access['selectedRouteGroup'].join(',')}`
            }
            if ((route_access).hasOwnProperty('selectedRouteRegion')) {
                base_access = `${base_access} > ${route_access['selectedRouteRegion'].join(',')}`
            }
            if ((route_access).hasOwnProperty('selectedRouteCountry')) {
                base_access = `${base_access} > ${route_access['selectedRouteCountry'].join(',')}`
            }
            if ((route_access).hasOwnProperty('selectedRoute')) {
                base_access = `${base_access} > ${route_access['selectedRoute'].join(',')}`
            }
        }
        return base_access;
    }

    actions(params) {
        var element = document.createElement("span");
        var icon1 = document.createElement("i");
        var icon2 = document.createElement("i");

        icon1.className = 'fa fa-pencil'
        icon1.onclick = () =>
            this.props.history.push({
                pathname: "/editUser",
                state: { data: params.data }
            });

        icon2.className = 'fa fa-trash'
        icon2.onclick = () => {
            Swal.fire({
                title: 'Warning!',
                text: 'Do you want to delete this user?',
                icon: 'warning',
                confirmButtonText: 'Yes',
                showCancelButton: true,
                cancelButtonText: 'No',
                cancelButtonColor: '#ff0000bf'
            }).then((result) => {
                if (result.value) {
                    this.deleteUser(params.data.id)
                }
            })
        }
        element.appendChild(icon1);
        element.appendChild(icon2);
        return element;
    }

    deleteUser = (id) => {
        api.delete(`rest/users/${id}/`)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `User deleted successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.getUserList();
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to delete user`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    handleChange = (e) => {
        this.setState({ [e.target.id]: e.target.value });
        if (e.target.value === '') {
            setTimeout(() => {
                this.getUserList()
            }, 1000);
        }
    }

    firstDataRendered = (params) => {
        params.api.sizeColumnsToFit();
    }

    applyColumnLockDown(columnDefs) {
        return columnDefs.map((c) => {
            if (c.children) {
                c.children = this.applyColumnLockDown(c.children);
            }
            return {
                ...c, ...{
                    lockPosition: true,
                    cellClass: 'lock-pinned'
                }
            }
        }
        )
    }

    onGridReady = (params) => {
        if (!params.api) {
            return null;
        }
        this.gridApi = params.api;
        // this.gridApi.sizeColumnsToFit();
    }

    render() {

        setInterval(() => {
            if (this.gridApi) {
                if (this.gridApi.gridPanel.eBodyViewport.clientWidth) {
                    this.gridApi.sizeColumnsToFit();
                }
            }
        }, 1000)

        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='user-module-search'>
                    <div className='add'>
                        <h2>Users</h2>
                        <button type="button" className="btn btn-primary" onClick={() => this.props.history.push('/createUser')}>ADD USER</button>
                    </div>
                    <div className='module-form'>
                        <div class="form-group">
                            <label for="userName">User Name</label>
                            <input type="text" class="form-control" id="userName" value={this.state.userName} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <div class="form-group">
                            <label for="email">Email</label>
                            <input type="text" class="form-control" id="email" value={this.state.email} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <div class="form-group">
                            <label for="role">Role</label>
                            <input type="text" class="form-control" id="role" value={this.state.role} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <button type="button" className="btn btn-danger" onClick={() => this.getUserList()}>Search</button>
                    </div>
                    <div className='user-module-table'>
                        {this.state.loading ?
                            <Spinners /> :
                            <div
                                id='myGrid' className={`ag-theme-balham-dark ag-grid-table`}
                                style={{ height: 'calc(100vh - 330px)' }}
                            >
                                <AgGridReact
                                    onGridReady={this.onGridReady}
                                    onFirstDataRendered={(params) => this.firstDataRendered(params)}
                                    columnDefs={this.applyColumnLockDown(this.state.userColumn)}
                                    rowData={this.state.userData}
                                    paginationPageSize={15}
                                    pagination={true}
                                    rowSelection={`single`}
                                >
                                </AgGridReact>
                            </div >}
                    </div>
                </div>
            </div>

        );
    }
}

export default SearchUser;