import React, { Component } from 'react';
import { Redirect } from "react-router-dom";
import api from '../../API/api'
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Access from "../../Constants/accessValidation";
import Loader from '../../Component/Loader';
import './systemModule.scss';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import '../../ag-grid-table.scss';
import Spinners from "../../spinneranimation";
import Swal from 'sweetalert2';


const apiServices = new APIServices();


class SystemModuleSearch extends Component {
    constructor(props) {
        super(props);
        this.gridApi = null;
        this.state = {
            moduleName: '',
            moduleType: '',
            moduleColumn: [
                { headerName: 'Id', field: 'id', width: 300 },
                { headerName: 'Module Name', field: 'Module_Name' },
                { headerName: 'Module Type', field: 'module_type' },
                { headerName: 'Actions', field: '', cellRenderer: (params) => this.actions(params), width: 250 }
            ],
            moduleData: [],
        };

    }

    componentDidMount() {
        if (Access.accessValidation('System Modules', 'systemModuleSearch')) {
            this.getSystemModulelist()
        } else {
            this.props.history.push('/404')
        }
    }

    actions(params) {
        var element = document.createElement("span");
        var icon1 = document.createElement("i");
        var icon2 = document.createElement("i");

        icon1.className = 'fa fa-pencil'
        icon1.onclick = () =>
            this.props.history.push({
                pathname: "/systemModuleEdit",
                state: { data: params.data }
            });

        icon2.className = 'fa fa-trash'
        icon2.onclick = () => {
            Swal.fire({
                title: 'Warning!',
                text: 'Do you want to delete this module?',
                icon: 'warning',
                confirmButtonText: 'Yes',
                showCancelButton: true,
                cancelButtonText: 'No',
                cancelButtonColor: '#ff0000bf'
            }).then((result) => {
                if (result.value) {
                    this.deleteSystemModule(params.data.id)
                }
            })
        }
        element.appendChild(icon1);
        element.appendChild(icon2);
        return element;
    }

    getSystemModulelist = () => {
        this.setState({ loading: true, moduleData: [] })
        const { moduleName, moduleType } = this.state;
        const params = {};
        if (moduleName) {
            params['name'] = moduleName
        }
        if (moduleType) {
            params['module_type'] = moduleType
        }

        const serialize = (params) => {
            var str = [];
            for (var p in params)
                if (params.hasOwnProperty(p)) {
                    str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
                }
            return str.join("&");
        }

        api.get(`sysmodwithsearch?${serialize(params)}`, 'hide')
            .then((response) => {
                this.setState({ loading: false })
                console.log(response, 'respo')
                if (response) {
                    if (response.data.response.length > 0) {
                        let moduleData = [];
                        response.data.response.forEach(function (key) {
                            moduleData.push({
                                'id': key.id,
                                'Module_Name': key.name,
                                'Module_Path': key.path,
                                'create_date': key.create_date,
                                'is_deleted': key.is_deleted,
                                'parentid': key.parentid,
                                'event_code': key.event_code,
                                'module_type': key.module_type,
                                'visibility': key.visibility
                            });
                        });
                        this.setState({ moduleData: moduleData });
                    }
                }
            })
            .catch((err) => {
                this.setState({ loading: false })
            })
    }

    deleteSystemModule = (id) => {
        api.delete(`rest/createmodules/${id}/`)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `System Module deleted successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    this.getSystemModulelist();
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to delete system module`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }


    handleChange = (e) => {
        this.setState({ [e.target.id]: e.target.value });
        if (e.target.value === '') {
            setTimeout(() => {
                this.getSystemModulelist()
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
                <div className='system-module-search'>
                    <div className='add'>
                        <h2>System Modules</h2>
                        <button type="button" className="btn btn-primary" onClick={() => this.props.history.push('/systemModuleAdd')}>ADD</button>
                    </div>
                    <div className='module-form'>
                        <div class="form-group">
                            <label for="moduleName">Module Name</label>
                            <input type="text" class="form-control" id="moduleName" value={this.state.moduleName} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <div class="form-group">
                            <label for="moduleType">Module Type</label>
                            <input type="text" class="form-control" id="moduleType" value={this.state.moduleType} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <button type="button" className="btn btn-danger" onClick={() => this.getSystemModulelist()}>Search</button>
                    </div>
                    <div className='system-module-table'>
                        {this.state.loading ?
                            <Spinners /> :
                            <div
                                id='myGrid' className={`ag-theme-balham-dark ag-grid-table`}
                                style={{ height: 'calc(100vh - 330px)' }}
                            >
                                <AgGridReact
                                    onGridReady={this.onGridReady}
                                    onFirstDataRendered={(params) => this.firstDataRendered(params)}
                                    columnDefs={this.applyColumnLockDown(this.state.moduleColumn)}
                                    rowData={this.state.moduleData}
                                    // onCellClicked={onCellClicked}
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

export default SystemModuleSearch;